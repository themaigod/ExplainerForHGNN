from .explainer import Explainer, ExplainerCore
from .explanation import NodeExplanationCombination, NodeExplanation
from .node_dataset_scores import node_dataset_scores
from .prepare_explanation_for_node_scores import standard_explanation
from .prepare_combined_explanation_for_node_dataset_scores import \
    prepare_combined_explanation_fn_for_node_dataset_scores
from .prepare_explanation_for_node_dataset_scores import \
    prepare_explanation_fn_for_node_dataset_scores
import torch


class GradExplainerCore(ExplainerCore):
    def __init__(self, config):
        super().__init__(config)

    def explain(self, model, **kwargs):
        self.model = model
        self.model.eval()

        if self.model.dataset.single_graph:
            self.node_id = kwargs.get('node_id', None)

        self.init_params()

        self.extract_neighbors_input()

        if self.model.dataset.single_graph:
            if self.node_id is None:
                raise ValueError('node_id is required for node-level explanation')
            return self.node_level_explain()
        else:
            return self.graph_level_explain()

    def init_params(self):
        pass

    def init_params_graph_level(self):
        pass

    def init_params_node_level(self):
        pass

    def graph_level_explain(self):
        pass

    def extract_neighbors_input(self):
        gs, features = self.model.standard_input()
        self.neighbor_input = {"gs": gs, "feature": features}
        return gs, features

    def node_level_explain(self):
        self.fit()
        return self.construct_explanation()

    def construct_explanation(self):
        explanation = NodeExplanation()
        explanation = standard_explanation(explanation, self)
        explanation.node_mask = self.node_mask
        for metric in self.config['eval_metrics']:
            prepare_explanation_fn_for_node_dataset_scores[metric](explanation, self)
        self.explanation = explanation
        return explanation

    def get_required_fit_params(self):
        pass

    def fit(self):
        if self.model.dataset.single_graph:
            self.fit_node_level()
        else:
            self.fit_graph_level()

    def fit_graph_level(self):
        pass

    def fit_node_level(self):
        loss = self.forward()
        loss.backward()
        features = self.temp_features
        if self.model.support_multi_features and self.config.get('use_meta', False):
            features_weight = [i.grad for i in features]
        else:
            features_weight = features.grad
        # normalize the weight
        if self.model.support_multi_features and self.config.get('use_meta', False):
            features_weight = [i / torch.sqrt(torch.sum(i ** 2, dim=1, keepdim=True)) for i in features_weight]
        else:
            features_weight = torch.sqrt(torch.sum(features_weight ** 2, dim=1))
        self.node_mask = features_weight

        self.edge_mask = self.convert_node_mask_to_edge_mask(self.node_mask, self.temp_gs)

    def convert_node_mask_to_edge_mask(self, node_mask, gs):
        edge_mask = []
        for g in gs:
            edge_mask.append(self.convert_node_mask_to_edge_mask_single(node_mask, g))
        return edge_mask

    @staticmethod
    def convert_node_mask_to_edge_mask_single(node_mask, g):
        g = g.coalesce()
        indices = g.indices()
        values = g.values()
        edge_mask = torch.zeros_like(values)
        for i in range(len(edge_mask)):
            edge_mask[i] = node_mask[indices[0][i]] * node_mask[indices[1][i]]
        return edge_mask

    def get_loss(self, output, mask=None):
        if self.model.dataset.single_graph:
            return self.get_loss_node_level(output, mask)
        else:
            return self.get_loss_graph_level(output, mask)

    def get_loss_graph_level(self, output, mask=None):
        pass

    def get_loss_node_level(self, output, mask=None):
        if mask is not None:
            class_id = mask
        else:
            class_id = output.argmax()
        loss = self.model.custom_loss(output, class_id)
        return loss

    def get_input_handle_fn(self):
        if self.model.dataset.single_graph:
            return self.get_input_handle_fn_node_level()
        else:
            return self.get_input_handle_fn_graph_level()

    def get_input_handle_fn_graph_level(self):
        pass

    def get_input_handle_fn_node_level(self):

        if self.model.support_multi_features and self.config.get('use_meta', False):
            def handle_fn(model):
                gs, features = model.standard_input()
                features_list = []
                for i in range(len(gs)):
                    features_list.append(features.clone().detach())
                for i in features_list:
                    i.requires_grad = True
                self.temp_features = features_list
                self.temp_gs = gs
                return gs, features_list

        else:
            def handle_fn(model):
                gs, features = model.standard_input()
                features.requires_grad = True
                self.temp_features = features
                self.temp_gs = gs
                return gs, features

        return handle_fn

    def forward(self):
        if self.model.dataset.single_graph:
            return self.forward_node_level()
        else:
            return self.forward_graph_level()

    def forward_graph_level(self):
        pass

    def forward_node_level(self):
        logits = self.model.custom_forward(self.get_input_handle_fn())
        if self.config.get('use_label', False):
            loss = self.get_loss(logits, self.model.labels[self.node_id])
        else:
            loss = self.get_loss(logits)
        return loss

    def build_optimizer(self):
        pass

    def build_scheduler(self, optimizer):
        pass

    def visualize(self):
        # !TODO: finish it, but not now
        pass

    @property
    def edge_mask_for_output(self):
        if 'edge_mask' not in self.__dict__:
            return None
        return self.edge_mask

    @property
    def feature_mask_for_output(self):
        return None

    def get_custom_input_handle_fn(self, masked_gs=None, feature_mask=None):
        """
        Get the custom input handle function for the model.
        :return:
        """

        def handle_fn(model):
            if model is None:
                model = self.model
            gs, features = model.standard_input()
            if masked_gs is not None:
                gs = [i.to(self.device_string) for i in masked_gs]
            if feature_mask is not None:
                feature_mask_device = feature_mask.to(self.device_string)
                features = features * feature_mask_device
            return gs, features

        return handle_fn

    @property
    def node_mask_for_output(self):
        if 'node_mask' not in self.__dict__:
            return None
        return self.node_mask


class GradExplainer(Explainer):
    def __init__(self, config):
        super().__init__(config)

    def explain(self, model, **kwargs):
        self.model = model

        if self.model.dataset.single_graph:
            self.node_level_explain(**kwargs)
        else:
            self.graph_level_explain(**kwargs)

    def node_level_explain(self, **kwargs):
        result = []
        test_labels = self.model.dataset.labels[2]

        if kwargs.get('max_nodes', None) is not None \
            and kwargs.get('max_nodes') < len(test_labels):
            test_labels = test_labels[:kwargs.get('max_nodes')]

        for idx, label in test_labels:
            explain_node = GradExplainerCore(self.config)
            explain_node.to(self.device)
            explanation = explain_node.explain(self.model, node_id=idx)
            result.append(explanation)

        result = self.construct_explanation(result)

        self.result = result

        self.evaluate()

        self.save_summary()

        return self.eval_result

    def explain_selected_nodes(self, model, selected_nodes):
        self.model = model
        result = []
        for idx in selected_nodes:
            if idx not in self.model.dataset.labels[2]:
                continue
            explain_node = GradExplainerCore(self.config)
            explain_node.to(self.device)
            explanation = explain_node.explain(self.model, node_id=idx)
            result.append(explanation)

        result = self.construct_explanation(result)

        self.result = result

        self.evaluate()

        self.save_summary()

        return self.eval_result

    def graph_level_explain(self, **kwargs):
        pass

    def construct_explanation(self, result):
        result = NodeExplanationCombination(node_explanations=result)
        if self.config.get('control_data', None) is not None:
            result.control_data = self.config['control_data']

        return result

    def evaluate(self):
        eval_result = {}
        if self.config.get('eval_metrics', None) is not None:
            for metric in self.config['eval_metrics']:
                # node_dataset_score_explanations_combined[metric](self.result, self)
                self.result = prepare_combined_explanation_fn_for_node_dataset_scores[
                    metric](self.result, self)
                eval_result[metric] = node_dataset_scores[metric](self.result)

        self.eval_result = eval_result
        return eval_result

    def get_summary(self):
        return self.eval_result

    def save_summary(self):
        if self.config.get('summary_path', None) is not None:
            import os
            os.makedirs(os.path.dirname(self.config['summary_path']),
                        exist_ok=True)
            import json
            with open(self.config['summary_path'], 'w') as f:
                json.dump(self.eval_result, f)

    def save_explanation(self, **kwargs):
        if self.config.get('explanation_path', None) is not None:
            import os
            os.makedirs(self.config['explanation_path'],
                        exist_ok=True)
            self.result.save(self.config['explanation_path'], **kwargs)


class GradExplainerOriginal(GradExplainer):
    def __init__(self, config):
        super().__init__(config)
        config['use_meta'] = False


class GradExplainerMeta(GradExplainer):
    def __init__(self, config):
        super().__init__(config)
        config['use_meta'] = True
