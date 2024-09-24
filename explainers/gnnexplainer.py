from .explainer import Explainer, ExplainerCore
import torch
import torch.nn.functional as fn
import torch.nn as nn
import math
# from .utils import node_scores, node_dataset_scores, standard_explanation, \
#     node_score_explanations, node_dataset_score_explanations, \
#     node_dataset_score_explanations_combined
from .node_scores import node_scores
from .node_dataset_scores import node_dataset_scores
from .prepare_explanation_for_node_scores import standard_explanation
from .prepare_combined_explanation_for_node_dataset_scores import \
    prepare_combined_explanation_fn_for_node_dataset_scores
from .prepare_explanation_for_node_scores import prepare_explanation_fn_for_node_scores
from .prepare_explanation_for_node_dataset_scores import \
    prepare_explanation_fn_for_node_dataset_scores
from .explanation import NodeExplanation, NodeExplanationCombination


def mean(iterable):
    return sum(iterable) / len(iterable)


class GNNExplainerMetaCore(ExplainerCore):
    def __init__(self, config):
        super().__init__(config)
        self.record_metrics = self.config.get('record_metrics', None)
        if not self.record_metrics:
            self.record_metrics = ['mask_density']

    def explain(self, model, **kwargs):
        self.model = model
        self.model.eval()

        self.init_params()

        if self.model.dataset.single_graph:
            self.node_id = kwargs.get('node_id', None)
            if self.node_id is None:
                raise ValueError('node_id is required for node-level explanation')
            return self.node_level_explain()
        else:
            return self.graph_level_explain()

    def graph_level_explain(self):
        pass

    def init_params(self):
        if self.model.dataset.single_graph:
            self.init_params_node_level()
        else:
            self.init_params_graph_level()
        self.registered_modules_and_params = {
            str(index): i for index, i in enumerate(self.get_required_fit_params())
        }
        self.to(self.device)

    def init_params_graph_level(self):
        pass

    def extract_neighbors_input(self):
        """
        Extract the neighbors of the node to be explained
        :return:
        """
        if self.config.get('extract_neighbors', False):
            gs, features = self.model.standard_input()
            self.neighbor_input = {"gs": gs, "features": features}
            return gs, features

        # if self.neighbor_input.get("gs", None) is not None:
        #     return self.neighbor_input["gs"], self.neighbor_input["features"]
        if getattr(self, 'neighbor_input',
                   None) is not None and self.neighbor_input.get(
            "gs", None) is not None:
            return self.neighbor_input["gs"], self.neighbor_input["features"]

        self.n_hop = self.config.get('n_hop', 3)  # default 3-hop neighbors

        gs, features = self.model.standard_input()

        used_nodes_set = set()

        for g in gs:
            indices = g.indices()
            # mask = indices[0] == self.node_id

            # consider memory-efficient
            current_nodes = [self.node_id]

            for i in range(self.n_hop):
                new_current_nodes = []
                for node in current_nodes:
                    mask = indices[0] == node
                    used_nodes_set.update(indices[1][mask].tolist())
                    new_current_nodes.extend(indices[1][mask].tolist())

                current_nodes = new_current_nodes

        self.used_nodes = sorted(list(used_nodes_set))
        self.recovery_dict = {node: i for i, node in enumerate(self.used_nodes)}

        # now reconstruct the graph
        new_gs = []
        for g in gs:
            indices = g.indices()
            # !TODO: maybe some problems here, not consider n_hop
            mask = (indices[0] == self.node_id) & (indices[1] == self.node_id)
            new_indices = torch.stack(
                [torch.tensor([self.recovery_dict[node] for node in indices[0][mask]]),
                 torch.tensor([self.recovery_dict[node] for node in indices[1][mask]])],
                dim=0)
            new_values = g.values()[mask]
            new_gs.append(torch.sparse_coo_tensor(new_indices, new_values, g.shape))

        self.neighbor_input = {"gs": new_gs, "features": features[self.used_nodes]}

    def init_params_node_level(self):
        # consider for heterogeneous graph
        # size is list (len(gs)) of number of edges in each graph

        gs, features = self.extract_neighbors_input()

        self.edge_mask = []

        for g in gs:
            self.edge_mask.append(torch.randn(g._nnz(), device=self.model.device))

        num_node = gs[0].shape[0]

        # self.edge_mask = [torch.nn.Parameter(em) for em in self.edge_mask]
        std = torch.nn.init.calculate_gain('relu') * math.sqrt(2.0 / (2 * num_node))
        if self.config['init_strategy_for_edge'] == 'normal':
            edge_mask_after_init = []
            for em in self.edge_mask:
                edge_mask_after_init.append(
                    torch.nn.Parameter(
                        torch.randn_like(em, device=self.model.device) * std))
            self.edge_mask = edge_mask_after_init


        elif self.config['init_strategy_for_edge'] == 'const':
            edge_mask_after_init = []
            for em in self.edge_mask:
                edge_mask_after_init.append(
                    torch.nn.Parameter(
                        torch.ones_like(em, device=self.model.device) * self.config[
                            'init_const']))
            self.edge_mask = edge_mask_after_init

        self.pram_feature_mask = nn.ParameterList(self.edge_mask)

        if self.config['use_mask_bias']:
            self.edge_mask_bias = []
            for g in gs:
                self.edge_mask_bias.append(
                    torch.nn.Parameter(torch.zeros(g._nnz(), device=self.model.device)))
                for em in self.edge_mask_bias:
                    torch.nn.init.constant_(em, 0.0)

            self.pram_feature_mask_bias = nn.ParameterList(self.edge_mask_bias)
        else:
            self.edge_mask_bias = None

        feat_dim = features.shape[1]
        mask = torch.randn(feat_dim, device=self.model.device)
        if self.config['init_strategy_for_feature'] == 'normal':
            with torch.no_grad():
                mask.normal_(1.0, 0.1)
        elif self.config['init_strategy_for_feature'] == 'const':
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
        self.pram_feature_mask = torch.nn.Parameter(mask)
        self.feature_mask = [self.pram_feature_mask]

    def node_level_explain(self):
        self.fit()
        return self.construct_explanation()

    def construct_explanation(self):
        explanation = NodeExplanation()
        explanation = standard_explanation(explanation, self)
        for metric in self.config['eval_metrics']:
            prepare_explanation_fn_for_node_dataset_scores[metric](explanation, self)
        self.explanation = explanation
        return explanation

    def get_required_fit_params(self):
        if self.edge_mask_bias is not None:
            return self.edge_mask + self.edge_mask_bias + self.feature_mask
        return self.edge_mask + self.feature_mask

    def fit(self):
        if self.model.dataset.single_graph:
            self.fit_node_level()
        else:
            self.fit_graph_level()

    def fit_graph_level(self):
        pass

    def fit_node_level(self):
        record = {}

        optimizer = self.build_optimizer()
        if self.config.get('opt_scheduler', None) is not None:
            scheduler = self.build_scheduler(optimizer)
        else:
            scheduler = None

        self.model.eval()

        for epoch in range(self.config['epochs']):
            optimizer.zero_grad()
            loss = self.forward()
            self.current_loss = loss.item()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if self.config['record']:
                if epoch % self.config['record_step'] == 0:
                    record[epoch] = {}
                    record[epoch]['loss'] = loss.item()
                    explanation = NodeExplanation()
                    for metric in self.record_metrics:
                        prepare_explanation_fn_for_node_scores[metric](explanation,
                                                                       self)
                        record[epoch][metric] = node_scores[metric](explanation)

                    output_str = 'Node: {} Epoch: {}, Loss: {:.4f}'.format(self.node_id,
                                                                           epoch,
                                                                           loss.item())
                    for metric in self.record_metrics:
                        output_str += ', {}: {:.4f}'.format(metric,
                                                            record[epoch][metric])
                    print(output_str)

        self.record = record

    def get_loss(self, output, mask=None):
        if self.model.dataset.single_graph:
            return self.get_loss_node_level(output, mask)
        else:
            return self.get_loss_graph_level(output, mask)

    def get_loss_graph_level(self, output, mask=None):
        pass

    def get_loss_node_level(self, output, mask=None):
        # we have several loss terms
        # normal loss term
        # edge mask size loss term
        # feature mask size loss term
        # edge mask entropy loss term
        # feature mask entropy loss term ## not sure about this term, other third party packages have this term but the official implementation implement without using
        # laplacian loss term ## not sure about this term, other third party packages do not have this term but the official implementation has
        # finally keep both feature mask entropy and laplacian
        if mask:
            normal_loss = self.model.custom_loss(output, mask)
        else:
            normal_loss = self.model.loss(output)

        if self.config['edge_mask_activation'] == 'sigmoid':
            edge_mask = [fn.sigmoid(em) for em in self.edge_mask]
            edge_mask_size_loss = [torch.sum(em) for em in edge_mask]
            edge_mask_size_loss = sum(edge_mask_size_loss)
        elif self.config['edge_mask_activation'] == 'relu':
            edge_mask = [fn.relu(em) for em in self.edge_mask]
            edge_mask_size_loss = [torch.sum(em) for em in edge_mask]
            edge_mask_size_loss = sum(edge_mask_size_loss)
        else:
            raise ValueError('edge mask activation {} not supported'.format(
                self.config['edge_mask_activation']))

        if self.config['feature_mask_use_sigmoid']:
            feature_mask = fn.sigmoid(self.feature_mask[0])
        else:
            feature_mask = self.feature_mask[0]
        feature_mask_size_loss = torch.mean(feature_mask)

        # mask_entropy_loss = torch.mean(-edge_mask * torch.log(edge_mask + 1e-6) - (1 - edge_mask) * torch.log(1 - edge_mask + 1e-6))

        edge_mask_entropy_loss = mean([torch.mean(
            -em * torch.log(em + 1e-6) - (1 - em) * torch.log(1 - em + 1e-6)) for em in
            edge_mask])

        feature_mask_entropy_loss = torch.mean(
            -feature_mask * torch.log(feature_mask + 1e-6) - (
                1 - feature_mask) * torch.log(1 - feature_mask + 1e-6))

        pred = self.model.custom_forward(self.get_input_handle_fn())
        # transform to hard pred label
        pred = torch.argmax(pred, dim=1)

        laplacian_loss_all = 0
        for g in self.masked['masked_gs']:
            g = g.coalesce()
            indices = g.indices()
            values = g.values()
            degree = torch.sparse.sum(g, dim=0).to_dense()

            L = torch.sparse_coo_tensor(g.indices(), -values, g.size())
            L = L + torch.sparse_coo_tensor(
                torch.stack([indices[0], indices[0]], dim=0), degree[indices[0]],
                g.size())

            pred = pred.float().view(-1, 1)
            pred_t = pred.t()
            laplacian_loss = (pred_t @ (torch.sparse.mm(L, pred))) / g._nnz()
            laplacian_loss_all += laplacian_loss

        laplacian_loss = laplacian_loss_all / len(self.masked['masked_gs'])

        loss = self.config['coff_normal'] * normal_loss + self.config[
            'coff_edge_size'] * edge_mask_size_loss + self.config[
                   'coff_feature_size'] * feature_mask_size_loss + self.config[
                   'coff_edge_ent'] * edge_mask_entropy_loss + self.config[
                   'coff_feature_ent'] * feature_mask_entropy_loss + self.config[
                   'coff_laplacian'] * laplacian_loss
        return loss

    def get_input_handle_fn(self):
        if self.model.dataset.single_graph:
            return self.get_input_handle_fn_node_level()
        else:
            return self.get_input_handle_fn_graph_level()

    def get_input_handle_fn_node_level(self):

        # ensure that each core module only has one
        masked = {}

        self.masked = masked

        def handle_fn(model):
            # directly use the computed neighbor graphs

            gs, features = self.extract_neighbors_input()

            if self.config['edge_mask_activation'] == 'sigmoid':
                sym_mask = [fn.sigmoid(em) for em in self.edge_mask]
            elif self.config['edge_mask_activation'] == 'relu':
                sym_mask = [fn.relu(em) for em in self.edge_mask]
            else:
                sym_mask = self.edge_mask

            masked_gs_generated = []
            for i in range(len(gs)):
                gs[i] = gs[i].coalesce()
                sym_mask_sparse = torch.sparse_coo_tensor(gs[i].indices(),
                                                          sym_mask[i],
                                                          gs[i].shape)
                masked_gs = gs[i] * sym_mask_sparse
                if self.config['use_mask_bias']:
                    edge_mask_bias_sparse = torch.sparse_coo_tensor(gs[i].indices(),
                                                                    self.edge_mask_bias[
                                                                        i],
                                                                    gs[i].shape)
                    edge_mask_bias_sparse = fn.relu6(
                        edge_mask_bias_sparse * 6) / 6  # not sure why, just following the original implementation
                    masked_gs = masked_gs + edge_mask_bias_sparse

                # remove the self-loop
                # here we use reconstruction this matrix to remove the self-loop
                masked_gs = masked_gs.coalesce()
                indices = masked_gs.indices()
                values = masked_gs.values()
                mask = indices[0] != indices[1]
                indices = indices[:, mask]
                values = values[mask]
                masked_gs = torch.sparse_coo_tensor(indices, values, gs[i].shape)

                masked_gs_generated.append(masked_gs)

            if self.config['mask_features']:
                feature_mask = fn.sigmoid(self.feature_mask[0]) if self.config[
                    'feature_mask_use_sigmoid'] else self.feature_mask[0]
                if self.config['feature_mask_marginalize']:
                    std_tensor = torch.ones_like(features, dtype=torch.float) / 2
                    mean_tensor = torch.zeros_like(features,
                                                   dtype=torch.float) - features
                    z = torch.normal(mean=mean_tensor, std=std_tensor)
                    features = features + z * (1 - feature_mask)
                else:
                    features = features * feature_mask

            masked['masked_gs'] = masked_gs_generated
            masked['masked_features'] = features

            return masked_gs_generated, features

        handle_fn(self.model)

        return handle_fn

    def get_input_handle_fn_graph_level(self):
        pass

    def forward(self):
        if self.model.dataset.single_graph:
            return self.forward_node_level()
        else:
            return self.forward_graph_level()

    def forward_graph_level(self):
        pass

    def forward_node_level(self):

        handle_fn = self.get_input_handle_fn()

        output = self.model.custom_forward(handle_fn)
        loss = self.get_loss(output, self.node_id)
        return loss

    def build_optimizer(self):
        import torch.optim as optim

        filter_fn = filter(lambda p: p.requires_grad, self.get_required_fit_params())

        if self.config['opt'] == 'adam':
            optimizer = optim.Adam(filter_fn, lr=self.config['opt_lr'], weight_decay=
            self.config['opt_wd'])
        elif self.config['opt'] == 'sgd':
            optimizer = optim.SGD(filter_fn, lr=self.config['opt_lr'], weight_decay=
            self.config['opt_wd'], momentum=0.95)
        elif self.config['opt'] == 'rmsprop':
            optimizer = optim.RMSprop(filter_fn, lr=self.config['opt_lr'], weight_decay=
            self.config['opt_wd'])
        elif self.config['opt'] == 'adagrad':
            optimizer = optim.Adagrad(filter_fn, lr=self.config['opt_lr'], weight_decay=
            self.config['opt_wd'])
        else:
            raise ValueError('Optimizer {} not supported'.format(self.config['opt']))
        return optimizer

    def build_scheduler(self, optimizer):
        import torch.optim as optim

        if self.config['opt_scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.config[
                'opt_decay_step'],
                                                  gamma=self.config['opt_decay_rate'])
        elif self.config['opt_scheduler'] == 'cos':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             T_max=self.config[
                                                                 'opt_restart'])
        else:
            raise ValueError(
                'Scheduler {} not supported'.format(self.config['opt_scheduler']))
        return scheduler

    def visualize(self):
        # !TODO: finish it, but not now
        pass

    @property
    def edge_mask_for_output(self):
        if self.config['edge_mask_activation'] == 'sigmoid':
            return [fn.sigmoid(em).clone().detach() for em in self.edge_mask]
        elif self.config['edge_mask_activation'] == 'relu':
            return [fn.relu(em).clone().detach() for em in self.edge_mask]
        else:
            return [em.clone().detach() for em in self.edge_mask]

    @property
    def feature_mask_for_output(self):
        if self.config['feature_mask_use_sigmoid']:
            return fn.sigmoid(self.feature_mask[0]).clone().detach()
        else:
            return self.feature_mask[0].clone().detach()

    def get_custom_input_handle_fn(self, masked_gs=None, feature_mask=None):
        """
        Get the custom input handle function for the model.
        :return:
        """

        def handle_fn(model):
            if model is None:
                model = self.model
            gs, features = self.extract_neighbors_input()
            if masked_gs is not None:
                gs = [i.to(self.device_string) for i in masked_gs]
            if feature_mask is not None:
                feature_mask_device = feature_mask.to(self.device_string)
                features = features * feature_mask_device
            return gs, features

        return handle_fn


class GNNExplainerMeta(Explainer):
    def __init__(self, config):
        super().__init__(config)

    def explain(self, model, **kwargs):
        self.model = model

        if self.model.dataset.single_graph:
            return self.node_level_explain()
        else:
            return self.graph_level_explain()

    def node_level_explain(self):

        result = []
        test_labels = self.model.dataset.labels[2]
        for idx, label in test_labels:
            explain_node = GNNExplainerMetaCore(self.config)
            explain_node.to(self.device)
            explanation = explain_node.explain(self.model,
                                               node_id=idx)
            result.append(explanation)

        # result = NodeExplanationCombination(node_explanations=result)

        result = self.construct_explanation(result)

        self.result = result

        self.evaluate()

        self.save_summary()

        return result

    def explain_selected_nodes(self, model, selected_nodes):
        self.model = model
        result = []
        test_labels = self.model.dataset.labels[2]
        for idx, label in test_labels:
            if idx in selected_nodes:
                explain_node = GNNExplainerMetaCore(self.config)
                explain_node.to(self.device)
                explanation = explain_node.explain(self.model,
                                                   node_id=idx)
                result.append(explanation)

        result = self.construct_explanation(result)

        self.result = result

        self.evaluate()

        self.save_summary()

        return result

    def construct_explanation(self, result):
        result = NodeExplanationCombination(node_explanations=result)
        if self.config.get('control_data', None) is not None:
            result.control_data = self.config['control_data']

        return result

    def graph_level_explain(self):
        pass

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
            import json
            with open(self.config['summary_path'], 'w') as f:
                json.dump(self.eval_result, f)

    def save_explanation(self):
        if self.config.get('explanation_path', None) is not None:
            import os
            os.makedirs(os.path.dirname(self.config['explanation_path']),
                        exist_ok=True)
            import json
            with open(self.config['explanation_path'], 'w') as f:
                json.dump(self.result.to_json(), f)


class GNNExplainerOriginalCore(Explainer, nn.Module):
    pass


class GNNExplainerOriginal(Explainer, nn.Module):
    pass
