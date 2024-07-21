import torch

"""
For a normal explanation, it usually contains the several attributes, which can be done by function:
standard_explanation(node_explanation, explainer)


This file contains the functions for the metrics in the explanation module.
There are two types of metrics:
1. node_scores: metrics for a single node explanation
2. node_dataset_scores: metrics for a dataset of node explanations

The metrics should be implemented as functions with the following signature:
def metric_name(node_explanation):
    return score

The metric functions should be registered in the corresponding dictionary:
1. node_scores: node_scores = {'metric_name': metric_name}
2. node_dataset_scores: node_dataset_scores = {'metric_name': metric_name}

Each metric should provide a process explanation function to access the necessary data for the metric calculation.
The process explanation function should be registered in the corresponding dictionary:
1. node_score_explanations: node_score_explanations = {'metric_name': process_explanation}
2. node_dataset_score_explanations: node_dataset_score_explanations = {'metric_name': process_explanation}

The process explanation function should have the following signature:
def metric_name_explanation(node_explanation, explainer):
    return node_explanation
    
Some metrics require additional hyperparameters to be calculated. These hyperparameters should be provided in the control_data.
Each metric should provide a process NodeExplanationCombination function to access these hyperparameters.
The process NodeExplanationCombination function should be registered in the corresponding dictionary:
1. node_dataset_score_explanations_combined: node_dataset_score_explanations_combined = {'metric_name': process_explanation}

The process NodeExplanationCombination function should have the following signature:
def metric_name_explanation_combined(node_explanations, explainer):
    return node_explanations
If metric_name_explanation_combined is not needed, it can be set to identity_explanation_combined.
"""


def standard_explanation(node_explanation, explainer):
    """
    standard explanation, introduced in GNNExplainer
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "edge_mask" not in node_explanation:
        edge_mask = explainer.edge_mask_for_output
        node_explanation.edge_mask = edge_mask

    if getattr(explainer, 'feature_mask_for_output',
               None) is not None and "feature_mask" not in node_explanation:
        feature_mask = explainer.feature_mask_for_output
        node_explanation.feature_mask = feature_mask

    if "node_id" not in node_explanation:
        node_id = explainer.node_id
        node_explanation.node_id = node_id

    return node_explanation


def mask_density(node_explanation):
    """
    explanation mask density, introduced in GNNExplainer
    :param node_explanation: a NodeExplanation object with edge_mask and neighbors_gs
    """

    gs = node_explanation.neighbors_gs
    edge_mask = node_explanation.edge_mask

    if isinstance(edge_mask, torch.Tensor):
        g = gs
        mask_density_score = mask_density_core(edge_mask, g)

    elif isinstance(edge_mask, list):
        mask_density_score = 0
        for mask, g in zip(edge_mask, gs):
            mask_density_score += mask_density_core(mask, g)
        mask_density_score /= len(edge_mask)
    else:
        raise ValueError('edge_mask should be a tensor or a list of tensors')

    return mask_density_score


def mask_density_core(edge_mask, g):
    """
    :param edge_mask: a tensor of shape (E,)
    :param g: a torch sparse coo tensor
    :return:
    """
    edge_mask = edge_mask.clone().detach()
    edge_mask[edge_mask < 0] = 0  # remove negative values

    mask_density_score = edge_mask.sum() / edge_mask.size(0)
    return mask_density_score.item()


node_scores = {
    'mask_density': mask_density,
}


def mask_density_explanation(node_explanation, explainer):
    """
    Set the edge_mask and neighbors_gs for the mask_density score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "neighbors_gs" not in node_explanation:
        neighbors_gs = explainer.neighbor_input['gs']
        node_explanation.neighbors_gs = neighbors_gs

    if "edge_mask" not in node_explanation:
        edge_mask = explainer.edge_mask_for_output
        node_explanation.edge_mask = edge_mask

    return node_explanation


node_score_explanations = {
    'mask_density': mask_density_explanation,
}


def fidelity_neg(node_explanations):
    """
    calculate the fidelity- score, based on the implementation in `torch_geometric.explain.metric.fidelity`
    introduced in <https://arxiv.org/abs/2206.09677>
    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    NodeExplanation object has the following attributes:
    label: ground truth label of the node
    masked_pred_label_hard: predicted label of the node after masking
    pred_label_hard: predicted label of the node
    :return:
    """
    masked_pred_labels_hard = node_explanations.pred_label_hard  # current type: list[tensor]
    labels = node_explanations.label  # current type: list[tensor]
    pred_labels_hard = node_explanations.masked_pred_label_hard  # current type: list[tensor]
    score = sum([fidelity_core(masked, label, pred) for masked, label, pred in
                 zip(masked_pred_labels_hard, labels, pred_labels_hard)]) / len(labels)
    return score


def fidelity_pos(node_explanations):
    """
    calculate the fidelity+ score, based on the implementation in `torch_geometric.explain.metric.fidelity`
    introduced in <https://arxiv.org/abs/2206.09677>
    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    NodeExplanation object has the following attributes:
    label: ground truth label of the node
    opposite_masked_pred_label_hard: predicted label of the node after masking by the opposite mask
    pred_label_hard: predicted label of the node
    :return:
    """
    opposite_masked_pred_labels_hard = node_explanations.opposite_masked_pred_label_hard  # current type: list[tensor]
    labels = node_explanations.label  # current type: list[tensor]
    pred_labels_hard = node_explanations.masked_pred_label_hard  # current type: list[tensor]
    score = sum([fidelity_core(masked, label, pred) for masked, label, pred in
                 zip(opposite_masked_pred_labels_hard, labels,
                     pred_labels_hard)]) / len(labels)
    return score


def fidelity_core(masked, label, pred):
    return abs(pred.eq(label).sum().item() - masked.eq(label).sum().item())


def characterization_score(node_explanations):
    """
    calculate the characterization score, based on the implementation in `torch_geometric.explain.metric.characterization`
    introduced in <https://arxiv.org/abs/2206.09677>
    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    NodeExplanationCombination object has the following key, value in control_data:
    pos_weight_characterization_score: the weight of the fidelity+ score, default is 0.5
    neg_weight_characterization_score: the weight of the fidelity- score, default is 0.5
    NodeExplanation object has the following attributes:
    label: ground truth label of the node
    masked_pred_label_hard: predicted label of the node after masking
    pred_label_hard: predicted label of the node
    :return:
    """
    fidelity_pos_score = fidelity_pos(node_explanations)
    fidelity_neg_score = fidelity_neg(node_explanations)
    pos_weight = node_explanations.control_data.get('pos_weight_characterization_score',
                                                    0.5)
    neg_weight = node_explanations.control_data.get('neg_weight_characterization_score',
                                                    0.5)
    if (pos_weight + neg_weight) != 1:
        raise ValueError('pos_weight + neg_weight should be 1')
    score = (fidelity_pos_score + fidelity_neg_score) / (
        pos_weight / fidelity_pos_score + neg_weight / fidelity_neg_score)
    return score


def fidelity_curve_auc(node_explanations):
    """
    calculate the AUC of the fidelity curve, based on the implementation in `torch_geometric.explain.metric.fidelity_curve_auc`
    introduced in <https://arxiv.org/abs/2206.09677>
    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    (Optional) NodeExplanationCombination object has the following key, value in control_data:
    fidelity_curve_auc_threshold: the threshold values to threshold the soft mask, like [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    NodeExplanation object has the following attributes:
    label: ground truth label of the node
    masked_pred_label_hard_threshold:
        a list of predicted labels of the node after masking while soft mask is thresholded by different values
        threshold values should be a list of threshold values to threshold the soft mask,
        and the distance between two adjacent threshold values will be treated as same (if fidelity_curve_auc_threshold is not given)
        Example:
        masked_pred_label_hard_threshold = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        # (not need to input) threshold_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    opposite_masked_pred_label_hard_threshold:
        a list of predicted labels of the node after masking by the opposite mask while soft mask is thresholded by different values
        threshold values should be a list of threshold values to threshold the soft mask,
        and the distance between two adjacent threshold values will be treated as same (if fidelity_curve_auc_threshold is not given)
        Example:
        opposite_masked_pred_label_hard_threshold = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        # (not need to input) threshold_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    pred_label_hard: predicted label of the node
    :return:
    """
    masked_pred_labels_hard_threshold = node_explanations.masked_pred_label_hard_threshold  # current type: list[list[tensor]]
    opposite_masked_pred_labels_hard_threshold = node_explanations.opposite_masked_pred_label_hard_threshold  # current type: list[list[tensor]]
    pred_labels_hard = node_explanations.pred_label_hard  # current type: list[tensor]
    labels = node_explanations.label  # current type: list[tensor]

    y = []
    for i in range(len(masked_pred_labels_hard_threshold[0])):
        masked_list = [masked_pred_labels_hard_threshold[j][i] for j in
                       range(len(masked_pred_labels_hard_threshold))]
        opposite_masked_list = [opposite_masked_pred_labels_hard_threshold[j][i] for j
                                in
                                range(len(opposite_masked_pred_labels_hard_threshold))]
        neg = sum([fidelity_core(masked, label, pred) for masked, label, pred in
                   zip(masked_list, labels, pred_labels_hard)]) / len(labels)
        pos = sum([fidelity_core(opposite_masked, label, pred) for
                   opposite_masked, label, pred in
                   zip(opposite_masked_list, labels, pred_labels_hard)]) / len(labels)
        score = pos / (1. - neg)
        y.append(score)

    if 'fidelity_curve_auc_threshold' in node_explanations.control_data:
        x = node_explanations.control_data['fidelity_curve_auc_threshold']
    else:
        x = [i / (len(y) - 1) for i in range(len(y))]

    # cover x, y to tensor
    x = torch.tensor(x)
    y = torch.tensor(y)
    auc = torch.trapezoid(y, x)
    return auc.item()


def unfaithfulness(node_explanations):
    """
    calculate the unfaithfulness score, based on the implementation in `torch_geometric.explain.metric.unfaithfulness`
    introduced in <https://arxiv.org/abs/2208.09339>
    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    NodeExplanation object has the following attributes:
    masked_pred_label: (soft) predicted label of the node after masking
    pred_label: (soft) predicted label of the node
    :return:
    """
    masked_pred_labels = node_explanations.masked_pred_label  # current type: list[tensor]
    pred_labels = node_explanations.pred_label  # current type: list[tensor]
    score = sum([unfaithfulness_core(masked, pred) for masked, pred in
                 zip(masked_pred_labels, pred_labels)]) / len(pred_labels)
    return score


def unfaithfulness_core(masked, pred):
    masked = masked.softmax(0)
    pred = pred.softmax(0)
    kl_div = torch.nn.functional.kl_div(pred.log(), masked, reduction='batchmean')
    return 1 - float(torch.exp(-kl_div))


def sparsity(node_explanations):
    """
    mask sparsity (similar to dig.xgraph.metric.sparsity)
    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    NodeExplanation object has the following attributes:
    edge_mask_hard: hard edge mask of the node
    :return:
    """
    edge_masks = node_explanations.edge_mask_hard  # current type: list[list[tensor]]
    sparsity_score = sum([sparsity_core(mask) for mask in edge_masks]) / len(edge_masks)
    return sparsity_score


def sparsity_core(edge_mask):
    total = sum(edge_mask[i].sum().item() for i in range(len(edge_mask)))
    return total / sum(edge_mask[i].size(0) for i in range(len(edge_mask)))


def graph_exp_faith_feature(node_explanations):
    """
    calculate the faithfulness score, based on the implementation in `GraphXAI.metrics.graph_exp_faith`

    Math:
    GEF = 1 - exp(-D_{KL}(P_{org} || P_{pert}))

    where:
    - D_{KL}(P_{org} || P_{pert}) = Σ P_{org}(i) log(P_{org}(i) / P_{pert}(i))
    - P_{org}: Softmax output of the model for the original graph.
    - P_{pert}: Softmax output of the model for the perturbed graph.

    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    NodeExplanation object has the following attributes:
    masked_pred_label: predicted label of the node after masking
    perturbed_feature_pred_label: predicted label of the node after perturbing the node feature
    :return:
    """
    masked_pred_labels = node_explanations.masked_pred_label  # current type: list[tensor]
    perturbed_feature_pred_labels = node_explanations.perturbed_feature_pred_label  # current type: list[tensor]
    score = sum([graph_exp_faith_core(masked, perturbed) for masked, perturbed in
                 zip(masked_pred_labels, perturbed_feature_pred_labels)]) / len(
        masked_pred_labels)
    return score


def graph_exp_faith_core(masked, perturbed):
    masked = masked.softmax(0)
    perturbed = perturbed.softmax(0)
    kl_div = torch.nn.functional.kl_div(masked.log(), perturbed, reduction='batchmean')
    return 1 - float(torch.exp(-kl_div))


def graph_exp_faith_edge(node_explanations):
    """
    calculate the faithfulness score, based on the implementation in `GraphXAI.metrics.graph_exp_faith`

    Math:
    GEF = 1 - exp(-D_{KL}(P_{org} || P_{pert}))

    where:
    - D_{KL}(P_{org} || P_{pert}) = Σ P_{org}(i) log(P_{org}(i) / P_{pert}(i))
    - P_{org}: Softmax output of the model for the original graph.
    - P_{pert}: Softmax output of the model for the perturbed graph.

    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    NodeExplanation object has the following attributes:
    masked_pred_label: predicted label of the node after masking
    perturbed_feature_pred_label: predicted label of the node after perturbing the edge feature
    :return:
    """
    masked_pred_labels = node_explanations.masked_pred_label  # current type: list[tensor]
    perturbed_feature_pred_labels = node_explanations.perturbed_feature_pred_label  # current type: list[tensor]
    score = sum([graph_exp_faith_core(masked, perturbed) for masked, perturbed in
                 zip(masked_pred_labels, perturbed_feature_pred_labels)]) / len(
        masked_pred_labels)
    return score


def graph_exp_stability_feature(node_explanations):
    """
    calculate the stability score, based on the implementation in `GraphXAI.metrics.graph_exp_stability`

    Math:
        GES = max(1 - (v_{ori} ⋅ v_{pert}) / (||v_{ori}|| ||v_{pert}||))

        where:
        - v_{ori}: Binary vector indicating the top k% important nodes in the original explanation.
        - v_{pert}: Binary vector indicating the top k% important nodes in the perturbed explanation.
        - ⋅ : Dot product operator.
        - ||v||: Magnitude of vector v.

    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    NodeExplanation object has the following attributes:
    feature_mask: feature mask of the node
    perturbed_feature_mask_for_stability: feature mask of the node after perturbing the node feature
        (it should be achieved several times, and we choose the best one, based on the GraphXAI implementation)
    :return:
    """
    feature_masks = node_explanations.feature_mask  # current type: list[tensor]
    perturbed_feature_masks = node_explanations.perturbed_feature_mask_for_stability  # current type: list[tensor]
    k = node_explanations.control_data.get('top_k_for_stability_feature', 0.25)
    score = sum([graph_exp_stability_core(feature_mask, perturbed_feature_mask, k) for
                 feature_mask, perturbed_feature_mask in
                 zip(feature_masks, perturbed_feature_masks)]) / len(feature_masks)
    return score


def graph_exp_stability_core(feature_mask, perturbed_feature_mask, k):
    feature_mask = feature_mask.flatten()
    top_k = int(k * len(feature_mask))
    top_k_indices = torch.argsort(feature_mask, descending=True)[:top_k]
    top_k_features = torch.zeros_like(feature_mask)
    top_k_features[top_k_indices] = 1
    scores = []
    for i in range(len(perturbed_feature_mask)):
        perturbed_top_k_indices = torch.argsort(perturbed_feature_mask[i],
                                                descending=True)[:top_k]
        perturbed_top_k_features = torch.zeros_like(feature_mask)
        perturbed_top_k_features[perturbed_top_k_indices] = 1
        score = 1 - torch.dot(top_k_features, perturbed_top_k_features) / (
            torch.norm(top_k_features) * torch.norm(perturbed_top_k_features)
        )
        scores.append(score)
    return max(scores)


def graph_exp_stability_edge(node_explanations):
    """
    calculate the stability score, based on the implementation in `GraphXAI.metrics.graph_exp_stability`

    Math:
        GES = max(1 - (v_{ori} ⋅ v_{pert}) / (||v_{ori}|| ||v_{pert}||))

        where:
        - v_{ori}: Binary vector indicating the top k% important nodes in the original explanation.
        - v_{pert}: Binary vector indicating the top k% important nodes in the perturbed explanation.
        - ⋅ : Dot product operator.
        - ||v||: Magnitude of vector v.

    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    NodeExplanation object has the following attributes:
    edge_mask: edge mask of the node
    perturbed_edge_mask_for_stability: edge mask of the node after perturbing the edge feature
        (it should be achieved several times, and we choose the best one, based on the GraphXAI implementation)
    :return:
    """
    edge_masks = node_explanations.edge_mask  # current type: list[tensor]
    perturbed_edge_masks = node_explanations.perturbed_edge_mask_for_stability  # current type: list[tensor]
    k = node_explanations.control_data.get('top_k_for_stability_edge', 0.25)
    score = sum([graph_exp_stability_core(edge_mask, perturbed_edge_mask, k) for
                 edge_mask, perturbed_edge_mask in
                 zip(edge_masks, perturbed_edge_masks)]) / len(edge_masks)
    return score


def MacroF1(node_explanations):
    masked_pred_labels_hard = node_explanations.masked_pred_label_hard  # current type: list[tensor]
    labels = node_explanations.label  # current type: list[tensor]
    score = f1_core(masked_pred_labels_hard, labels, average='macro')
    return score


def f1_core(masked, label, average='macro'):
    masked = [masked[i].item() for i in range(len(masked))]
    label = [label[i].item() for i in range(len(label))]
    from sklearn.metrics import f1_score
    return f1_score(label, masked, average=average)


def MicroF1(node_explanations):
    masked_pred_labels_hard = node_explanations.masked_pred_label_hard  # current type: list[tensor]
    labels = node_explanations.label  # current type: list[tensor]
    score = f1_core(masked_pred_labels_hard, labels, average='micro')
    return score


def roc_auc_score(node_explanations):
    masked_pred_labels_hard = node_explanations.masked_pred_label_hard  # current type: list[tensor]
    labels = node_explanations.label  # current type: list[tensor]
    score = roc_auc_core(masked_pred_labels_hard, labels)
    return score


def roc_auc_core(masked, label):
    masked = [masked[i].item() for i in range(len(masked))]
    label = [label[i].item() for i in range(len(label))]
    from sklearn.metrics import roc_auc_score as roc_auc_score_sklearn
    return roc_auc_score_sklearn(label, masked)


node_dataset_scores = {
    'fidelity_neg': fidelity_neg,
    'fidelity_pos': fidelity_pos,
    'characterization_score': characterization_score,
    'fidelity_curve_auc': fidelity_curve_auc,
    'unfaithfulness': unfaithfulness,
    'sparsity': sparsity,
    'graph_exp_faith_feature': graph_exp_faith_feature,
    'graph_exp_faith_edge': graph_exp_faith_edge,
    'graph_exp_stability_feature': graph_exp_stability_feature,
    'graph_exp_stability_edge': graph_exp_stability_edge,
    'Macro-F1': MacroF1,
    'Micro-F1': MicroF1,
    'roc_auc_score': roc_auc_score
}


def get_masked_gs_hard_threshold(gs, edge_mask, threshold, opposite=False):
    masked_gs_hard = []
    for g, em in zip(gs, edge_mask):
        values = g.values()
        mask = em > threshold if not opposite else em <= threshold
        masked_g = torch.sparse_coo_tensor(g.indices()[mask], values[mask], g.size())
        masked_gs_hard.append(masked_g)

    return masked_gs_hard


def get_feature_mask_hard_threshold(feature_mask, threshold, opposite=False):
    feature_mask_hard = feature_mask[0] > threshold if not opposite else feature_mask[
                                                                             0] <= threshold
    return feature_mask_hard


def fidelity_neg_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label_hard and pred_label_hard for the fidelity- score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label_hard" not in node_explanation:
        if "masked_pred_label" not in node_explanation:
            if "masked_gs_hard" not in node_explanation:
                gs = explainer.neighbor_input['gs']
                masked_gs_hard = get_masked_gs_hard_threshold(gs,
                                                              explainer.edge_mask_for_output,
                                                              explainer.config[
                                                                  'edge_mask_threshold'])
                node_explanation.masked_gs_hard = masked_gs_hard

            if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                       'feature_mask_for_output',
                                                                       None) is not None:
                feature_mask_hard = get_feature_mask_hard_threshold(
                    explainer.feature_mask_for_output,
                    explainer.config[
                        'feature_mask_threshold'])
                node_explanation.feature_mask_hard = feature_mask_hard

            masked_pred_label = \
                explainer.model.custom_forward(explainer.get_hard_input_handle_fn(
                    node_explanation.masked_gs_hard,
                    node_explanation.feature_mask_hard))[
                    explainer.node_id]
            node_explanation.masked_pred_label = masked_pred_label

        masked_pred_label_hard = node_explanation.masked_pred_label.argmax()
        node_explanation.masked_pred_label_hard = masked_pred_label_hard

    if "pred_label_hard" not in node_explanation:
        if "pred_label" not in node_explanation:
            pred_label = explainer.model()[explainer.node_id]
            node_explanation.pred_label = pred_label

        pred_label_hard = node_explanation.pred_label.argmax()
        node_explanation.pred_label_hard = pred_label_hard

    return node_explanation


def fidelity_pos_explanation(node_explanation, explainer):
    """
    Set the opposite_masked_pred_label_hard and opposite_masked_pred_label for the fidelity+ score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "opposite_masked_pred_label_hard" not in node_explanation:
        if "opposite_masked_pred_label" not in node_explanation:
            if "opposite_masked_gs_hard" not in node_explanation:
                gs = explainer.neighbor_input['gs']
                opposite_masked_gs_hard = get_masked_gs_hard_threshold(gs,
                                                                       explainer.edge_mask_for_output,
                                                                       explainer.config[
                                                                           'edge_mask_threshold'],
                                                                       opposite=True)
                node_explanation.opposite_masked_gs_hard = opposite_masked_gs_hard

            if "opposite_feature_mask_hard" not in node_explanation and getattr(
                explainer, 'feature_mask_for_output', None) is not None:
                opposite_feature_mask_hard = get_feature_mask_hard_threshold(
                    explainer.feature_mask_for_output,
                    explainer.config[
                        'feature_mask_threshold'],
                    opposite=True)
                node_explanation.opposite_feature_mask_hard = opposite_feature_mask_hard

            opposite_masked_pred_label = \
                explainer.model.custom_forward(explainer.get_hard_input_handle_fn(
                    node_explanation.opposite_masked_gs_hard,
                    node_explanation.opposite_feature_mask_hard))[
                    explainer.node_id]
            node_explanation.opposite_masked_pred_label = opposite_masked_pred_label

        opposite_masked_pred_label_hard = node_explanation.opposite_masked_pred_label.argmax()
        node_explanation.opposite_masked_pred_label_hard = opposite_masked_pred_label_hard

    if "pred_label_hard" not in node_explanation:
        if "pred_label" not in node_explanation:
            pred_label = explainer.model()[explainer.node_id]
            node_explanation.pred_label = pred_label

        pred_label_hard = node_explanation.pred_label.argmax()
        node_explanation.pred_label_hard = pred_label_hard

    return node_explanation


def characterization_score_explanation(node_explanation, explainer):
    """
    Set the necessary data for the characterization score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """

    node_explanation = fidelity_pos_explanation(node_explanation, explainer)
    node_explanation = fidelity_neg_explanation(node_explanation, explainer)

    return node_explanation


def fidelity_curve_auc_explanation(node_explanation, explainer):
    """
    Set masked_pred_label_hard_threshold, opposite_masked_pred_label_hard_threshold,
    pred_label_hard and label for the fidelity curve AUC score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label_hard_threshold" not in node_explanation or "opposite_masked_pred_label_hard_threshold" not in node_explanation:
        if "fidelity_curve_auc_threshold" in explainer.config:
            threshold_values = explainer.config['fidelity_curve_auc_threshold']
        else:
            num = 11
            threshold_values = [i / (num - 1) for i in range(num)]

        masked_pred_label_hard_threshold = []
        opposite_masked_pred_label_hard_threshold = []
        for threshold in threshold_values:
            if "masked_pred_label_hard_threshold" not in node_explanation:
                gs = explainer.neighbor_input['gs']
                masked_gs_hard = get_masked_gs_hard_threshold(gs,
                                                              explainer.edge_mask_for_output,
                                                              threshold)
                masked_pred_label = explainer.model.custom_forward(
                    explainer.get_hard_input_handle_fn(masked_gs_hard))[
                    explainer.node_id]
                masked_pred_label_hard_threshold.append(masked_pred_label.argmax())

            if "opposite_masked_pred_label_hard_threshold" not in node_explanation:
                gs = explainer.neighbor_input['gs']
                opposite_masked_gs_hard = get_masked_gs_hard_threshold(gs,
                                                                       explainer.edge_mask_for_output,
                                                                       threshold,
                                                                       opposite=True)
                opposite_masked_pred_label = explainer.model.custom_forward(
                    explainer.get_hard_input_handle_fn(opposite_masked_gs_hard))[
                    explainer.node_id]
                opposite_masked_pred_label_hard_threshold.append(
                    opposite_masked_pred_label.argmax())

        if "masked_pred_label_hard_threshold" not in node_explanation:
            node_explanation.masked_pred_label_hard_threshold = masked_pred_label_hard_threshold

        if "opposite_masked_pred_label_hard_threshold" not in node_explanation:
            node_explanation.opposite_masked_pred_label_hard_threshold = opposite_masked_pred_label_hard_threshold

    if "pred_label_hard" not in node_explanation:
        if "pred_label" not in node_explanation:
            pred_label = explainer.model()[explainer.node_id]
            node_explanation.pred_label = pred_label

        pred_label_hard = node_explanation.pred_label.argmax()
        node_explanation.pred_label_hard = pred_label_hard

    if "label" not in node_explanation:
        label_test = explainer.dataset.labels[2]
        label = None
        for idx, l in label_test:
            if idx == explainer.node_id:
                label = l
                break

        label = torch.tensor(label, device=explainer.model.device)

        node_explanation.label = label

    return node_explanation


def unfaithfulness_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label and pred_label for the unfaithfulness score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label" not in node_explanation:
        if "masked_gs_hard" not in node_explanation:
            gs = explainer.neighbor_input['gs']
            masked_gs_hard = get_masked_gs_hard_threshold(gs,
                                                          explainer.edge_mask_for_output,
                                                          explainer.config[
                                                              'edge_mask_threshold'])
            node_explanation.masked_gs_hard = masked_gs_hard

        if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                   'feature_mask_for_output',
                                                                   None) is not None:
            feature_mask_hard = get_feature_mask_hard_threshold(
                explainer.feature_mask_for_output,
                explainer.config[
                    'feature_mask_threshold'])
            node_explanation.feature_mask_hard = feature_mask_hard

        masked_pred_label = \
            explainer.model.custom_forward(explainer.get_hard_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.feature_mask_hard))[
                explainer.node_id]
        node_explanation.masked_pred_label = masked_pred_label

    if "pred_label" not in node_explanation:
        pred_label = explainer.model()[explainer.node_id]
        node_explanation.pred_label = pred_label

    return node_explanation


def sparsity_explanation(node_explanation, explainer):
    """
    Set the edge_mask for the sparsity score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "edge_mask_hard" not in node_explanation:
        if "edge_mask" not in node_explanation:
            edge_mask = explainer.edge_mask_for_output
            node_explanation.edge_mask = edge_mask

        edge_mask_hard = node_explanation.edge_mask > explainer.config[
            'edge_mask_threshold']
        node_explanation.edge_mask_hard = edge_mask_hard

    return node_explanation


def graph_exp_faith_feature_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label and perturbed_feature_pred_label for the faithfulness score

    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label" not in node_explanation:
        if "masked_gs_hard" not in node_explanation:
            gs = explainer.neighbor_input['gs']
            masked_gs_hard = get_masked_gs_hard_threshold(gs,
                                                          explainer.edge_mask_for_output,
                                                          explainer.config[
                                                              'edge_mask_threshold'])
            node_explanation.masked_gs_hard = masked_gs_hard

        if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                   'feature_mask_for_output',
                                                                   None) is not None:
            feature_mask_hard = get_feature_mask_hard_threshold(
                explainer.feature_mask_for_output,
                explainer.config[
                    'feature_mask_threshold'])
            node_explanation.feature_mask_hard = feature_mask_hard

        masked_pred_label = \
            explainer.model.custom_forward(explainer.get_hard_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.feature_mask_hard))[
                explainer.node_id]
        node_explanation.masked_pred_label = masked_pred_label

    if "perturbed_feature_pred_label" not in node_explanation:
        if "perturbed_feature_mask" not in node_explanation:
            perturbed_feature_mask = generate_perturbed_feature_mask(
                explainer.feature_mask_for_output,
                explainer.config['top_k_for_faith_feature'])
            node_explanation.perturbed_feature_mask = perturbed_feature_mask

        perturbed_feature_pred_label = \
            explainer.model.custom_forward(explainer.get_hard_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.perturbed_feature_mask))[
                explainer.node_id]
        node_explanation.perturbed_feature_pred_label = perturbed_feature_pred_label

    return node_explanation


def generate_perturbed_feature_mask(feature_mask, k):
    feature_mask = feature_mask[0]
    top_k = int(k * len(feature_mask))
    top_k_indices = torch.argsort(feature_mask, descending=True)[top_k:]
    perturbed_feature_mask = torch.zeros_like(feature_mask)
    perturbed_feature_mask[top_k_indices] = torch.normal(0, 0.1,
                                                         size=(len(top_k_indices),))
    return perturbed_feature_mask


def graph_exp_faith_edge_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label and perturbed_feature_pred_label for the faithfulness score

    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label" not in node_explanation:
        if "masked_gs_hard" not in node_explanation:
            gs = explainer.neighbor_input['gs']
            masked_gs_hard = get_masked_gs_hard_threshold(gs,
                                                          explainer.edge_mask_for_output,
                                                          explainer.config[
                                                              'edge_mask_threshold'])
            node_explanation.masked_gs_hard = masked_gs_hard

        if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                   'feature_mask_for_output',
                                                                   None) is not None:
            feature_mask_hard = get_feature_mask_hard_threshold(
                explainer.feature_mask_for_output,
                explainer.config[
                    'feature_mask_threshold'])
            node_explanation.feature_mask_hard = feature_mask_hard

        masked_pred_label = \
            explainer.model.custom_forward(explainer.get_hard_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.feature_mask_hard))[
                explainer.node_id]
        node_explanation.masked_pred_label = masked_pred_label

    if "perturbed_feature_pred_label" not in node_explanation:
        if "perturbed_edge_mask" not in node_explanation:
            perturbed_edge_mask = generate_perturbed_edge_mask(
                explainer.edge_mask_for_output,
                explainer.config[
                    'top_k_for_faith_edge'])
            node_explanation.perturbed_edge_mask = perturbed_edge_mask

        perturbed_feature_pred_label = \
            explainer.model.custom_forward(explainer.get_hard_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.perturbed_edge_mask))[
                explainer.node_id]
        node_explanation.perturbed_feature_pred_label = perturbed_feature_pred_label

    return node_explanation


def generate_perturbed_edge_mask(edge_mask, k):
    edge_mask = edge_mask[0]
    top_k = int(k * len(edge_mask))
    top_k_indices = torch.argsort(edge_mask, descending=True)[:top_k]
    perturbed_edge_mask = torch.zeros_like(edge_mask)
    perturbed_edge_mask[top_k_indices] = 1
    return perturbed_edge_mask


def graph_exp_stability_feature_explanation(node_explanation, explainer):
    """
    Set the feature_mask, feature_mask_hard and perturbed_feature_mask_for_stability for the stability score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "feature_mask" not in node_explanation:
        feature_mask = explainer.feature_mask_for_output
        node_explanation.feature_mask = feature_mask

    if "feature_mask_hard" not in node_explanation:
        feature_mask_hard = get_feature_mask_hard_threshold(
            node_explanation.feature_mask,
            explainer.config[
                'feature_mask_threshold'])
        node_explanation.feature_mask_hard = feature_mask_hard

    if "perturbed_feature_mask_for_stability" not in node_explanation:
        perturbed_feature_mask_for_stability = generate_perturbed_feature_mask_for_stability(
            explainer.feature_mask_for_output,
            explainer.config['top_k_for_stability_feature'])
        node_explanation.perturbed_feature_mask_for_stability = perturbed_feature_mask_for_stability

    return node_explanation


def generate_perturbed_feature_mask_for_stability(feature_mask, k):
    feature_mask = feature_mask[0]
    top_k = int(k * len(feature_mask))
    top_k_indices = torch.argsort(feature_mask, descending=True)[:top_k]
    perturbed_feature_mask = torch.zeros_like(feature_mask)
    perturbed_feature_mask[top_k_indices] = 1
    return perturbed_feature_mask


def graph_exp_stability_edge_explanation(node_explanation, explainer):
    """
    Set the edge_mask, edge_mask_hard and perturbed_edge_mask_for_stability for the stability score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "edge_mask" not in node_explanation:
        edge_mask = explainer.edge_mask_for_output
        node_explanation.edge_mask = edge_mask

    if "edge_mask_hard" not in node_explanation:
        edge_mask_hard = [i > explainer.config[
            'edge_mask_threshold'] for i in node_explanation.edge_mask]
        node_explanation.edge_mask_hard = edge_mask_hard

    if "perturbed_edge_mask_for_stability" not in node_explanation:
        perturbed_edge_mask_for_stability = generate_perturbed_edge_mask_for_stability(
            explainer.edge_mask_for_output,
            explainer.config['top_k_for_stability_edge'])
        node_explanation.perturbed_edge_mask_for_stability = perturbed_edge_mask_for_stability

    return node_explanation


def generate_perturbed_edge_mask_for_stability(edge_mask, k):
    edge_mask = edge_mask[0]
    top_k = int(k * len(edge_mask))
    top_k_indices = torch.argsort(edge_mask, descending=True)[:top_k]
    perturbed_edge_mask = torch.zeros_like(edge_mask)
    perturbed_edge_mask[top_k_indices] = 1
    return perturbed_edge_mask


def MacroF1_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label_hard and label for the Macro-F1 score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label_hard" not in node_explanation:
        if "masked_pred_label" not in node_explanation:
            if "masked_gs_hard" not in node_explanation:
                gs = explainer.neighbor_input['gs']
                masked_gs_hard = get_masked_gs_hard_threshold(gs,
                                                              explainer.edge_mask_for_output,
                                                              explainer.config[
                                                                  'edge_mask_threshold'])
                node_explanation.masked_gs_hard = masked_gs_hard

            if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                       'feature_mask_for_output',
                                                                       None) is not None:
                feature_mask_hard = get_feature_mask_hard_threshold(
                    explainer.feature_mask_for_output,
                    explainer.config[
                        'feature_mask_threshold'])
                node_explanation.feature_mask_hard = feature_mask_hard

            masked_pred_label = \
                explainer.model.custom_forward(explainer.get_hard_input_handle_fn(
                    node_explanation.masked_gs_hard,
                    node_explanation.feature_mask_hard))[
                    explainer.node_id]
            node_explanation.masked_pred_label = masked_pred_label

        masked_pred_label_hard = node_explanation.masked_pred_label.argmax()
        node_explanation.masked_pred_label_hard = masked_pred_label_hard

    if "label" not in node_explanation:
        label_test = explainer.dataset.labels[2]
        label = None
        for idx, l in label_test:
            if idx == explainer.node_id:
                label = l
                break

        label = torch.tensor(label, device=explainer.model.device)

        node_explanation.label = label

    return node_explanation


def MicroF1_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label_hard and label for the Micro-F1 score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label_hard" not in node_explanation:
        if "masked_pred_label" not in node_explanation:
            if "masked_gs_hard" not in node_explanation:
                gs = explainer.neighbor_input['gs']
                masked_gs_hard = get_masked_gs_hard_threshold(gs,
                                                              explainer.edge_mask_for_output,
                                                              explainer.config[
                                                                  'edge_mask_threshold'])
                node_explanation.masked_gs_hard = masked_gs_hard

            if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                       'feature_mask_for_output',
                                                                       None) is not None:
                feature_mask_hard = get_feature_mask_hard_threshold(
                    explainer.feature_mask_for_output,
                    explainer.config[
                        'feature_mask_threshold'])
                node_explanation.feature_mask_hard = feature_mask_hard

            masked_pred_label = \
                explainer.model.custom_forward(explainer.get_hard_input_handle_fn(
                    node_explanation.masked_gs_hard,
                    node_explanation.feature_mask_hard))[
                    explainer.node_id]
            node_explanation.masked_pred_label = masked_pred_label

        masked_pred_label_hard = node_explanation.masked_pred_label.argmax()
        node_explanation.masked_pred_label_hard = masked_pred_label_hard

    if "label" not in node_explanation:
        label_test = explainer.dataset.labels[2]
        label = None
        for idx, l in label_test:
            if idx == explainer.node_id:
                label = l
                break

        label = torch.tensor(label, device=explainer.model.device)

        node_explanation.label = label

    return node_explanation


node_dataset_score_explanations = {
    'fidelity_neg': fidelity_neg_explanation,
    'fidelity_pos': fidelity_pos_explanation,
    'characterization_score': characterization_score_explanation,
    'fidelity_curve_auc': fidelity_curve_auc_explanation,
    'unfaithfulness': unfaithfulness_explanation,
    'sparsity': sparsity_explanation,
    'graph_exp_faith_feature': graph_exp_faith_feature_explanation,
    'graph_exp_faith_edge': graph_exp_faith_edge_explanation,
    'graph_exp_stability_feature': graph_exp_stability_feature_explanation,
    'graph_exp_stability_edge': graph_exp_stability_edge_explanation,
    'Macro-F1': MacroF1_explanation,
    'Micro-F1': MicroF1_explanation
}


def identity_explanation_combined(node_explanations, explainer):
    """
    identity explanation, which does not change the input node_explanations
    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    :param explainer: an Explainer object
    """
    return node_explanations


def characterization_score_explanation_combined(node_explanations, explainer):
    """
    explanation for the characterization score, introduced in GNNExplainer
    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    :param explainer: an Explainer object
    """
    if 'pos_weight_characterization_score' not in node_explanations.control_data:
        pos_weight = explainer.config['pos_weight_characterization_score']
        node_explanations.control_data.update(
            {'pos_weight_characterization_score': pos_weight})
    if 'neg_weight_characterization_score' not in node_explanations.control_data:
        neg_weight = explainer.config['neg_weight_characterization_score']
        node_explanations.control_data.update(
            {'neg_weight_characterization_score': neg_weight})
    return node_explanations


def fidelity_curve_auc_explanation_combined(node_explanations, explainer):
    """
    explanation for the AUC of the fidelity curve, introduced in GNNExplainer
    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    :param explainer: an Explainer object
    """
    if "fidelity_curve_auc_threshold" not in node_explanations.control_data:
        if "fidelity_curve_auc_threshold" in explainer.config:
            threshold_values = explainer.config['fidelity_curve_auc_threshold']
        else:
            num = 11
            threshold_values = [i / (num - 1) for i in range(num)]
        node_explanations.control_data.update(
            {'fidelity_curve_auc_threshold': threshold_values})
    return node_explanations


def graph_exp_stability_feature_explanation_combined(node_explanations, explainer):
    """
    Set top_k_for_stability_feature in the control_data of the node_explanations

    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    :param explainer: an Explainer object
    """
    if 'top_k_for_stability_feature' not in node_explanations.control_data:
        if 'top_k_for_stability_feature' in explainer.config:
            k = explainer.config['top_k_for_stability_feature']
        else:
            k = 0.25
        node_explanations.control_data.update({'top_k_for_stability_feature': k})
    return node_explanations


def graph_exp_stability_edge_explanation_combined(node_explanations, explainer):
    """
    Set top_k_for_stability_edge in the control_data of the node_explanations

    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    :param explainer: an Explainer object
    """
    if 'top_k_for_stability_edge' not in node_explanations.control_data:
        if 'top_k_for_stability_edge' in explainer.config:
            k = explainer.config['top_k_for_stability_edge']
        else:
            k = 0.25
        node_explanations.control_data.update({'top_k_for_stability_edge': k})
    return node_explanations


node_dataset_score_explanations_combined = {
    'fidelity_neg': identity_explanation_combined,
    'fidelity_pos': identity_explanation_combined,
    'characterization_score': characterization_score_explanation_combined,
    'fidelity_curve_auc': fidelity_curve_auc_explanation_combined,
    'unfaithfulness': identity_explanation_combined,
    'sparsity': identity_explanation_combined,
    'graph_exp_faith_feature': identity_explanation_combined,
    'graph_exp_faith_edge': identity_explanation_combined,
    'graph_exp_stability_feature': graph_exp_stability_feature_explanation_combined,
    'graph_exp_stability_edge': graph_exp_stability_edge_explanation_combined,
    'Macro-F1': identity_explanation_combined,
    'Micro-F1': identity_explanation_combined,
    'roc_auc_score': identity_explanation_combined
}


def get_edge_mask_hard(explainer):
    if explainer.config['edge_mask_hard_method'] == 'threshold':
        edge_mask = explainer.edge_mask_for_output
        threshold = explainer.config['edge_mask_threshold']
        return [em > threshold for em in edge_mask]
    elif explainer.config['edge_mask_hard_method'] == 'auto_threshold':
        edge_mask = explainer.edge_mask_for_output
        edge_mask_concat = torch.cat(edge_mask)
        threshold = torch.quantile(edge_mask_concat,
                                   explainer.config['threshold_percentage'])
        return [em > threshold for em in edge_mask]
    elif explainer.config['edge_mask_hard_method'] == 'original':
        return explainer.edge_mask_for_output
    elif explainer.config['edge_mask_hard_method'] == 'top_k':
        edge_mask = explainer.edge_mask_for_output
        edge_mask_concat = torch.cat(edge_mask)
        top_k = int(explainer.config['top_k_for_edge_mask'] * len(edge_mask_concat))
        indices = torch.sort(edge_mask_concat, descending=True)[1][top_k]
        # revert the indices to the original indices
        indices_ori = []
        size_edge_mask = [em.size(0) for em in edge_mask]
        size_edge_mask_sum = [sum(size_edge_mask[:i]) for i in
                              range(len(size_edge_mask))]
        for i in indices:
            for j in range(len(size_edge_mask_sum)):
                if i >= size_edge_mask_sum[j]:
                    continue
                indices_ori.append((j, i - size_edge_mask_sum[j]))
                break

        edge_mask_hard = []
        for i in range(len(edge_mask)):
            edge_mask_hard.append(torch.zeros_like(edge_mask[i]))

        for i, j in indices_ori:
            edge_mask_hard[i][j] = 1

        return edge_mask_hard
    else:
        raise ValueError('Invalid edge_mask_threshold_method: {}'.format(
            explainer.config['edge_mask_threshold_method']))


def get_masked_gs_hard(explainer):
    if explainer.config['edge_mask_hard_method'] == 'original':
        return explainer.masked['masked_gs']
    edge_mask_hard = get_edge_mask_hard(explainer)
    gs = explainer.neighbor_input['gs']
    masked_gs_hard = []
    for g, em in zip(gs, edge_mask_hard):
        if explainer.config['edge_mask_hard_method'] != 'original':
            g = g.coalesce()
            values = g.values()
            mask = em
            masked_g = torch.sparse_coo_tensor(g.indices()[mask], values[mask],
                                               g.size())
            masked_gs_hard.append(masked_g)

    return masked_gs_hard
