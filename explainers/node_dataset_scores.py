import torch


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
    masked_pred_labels_hard = node_explanations.masked_pred_label_hard  # current type: list[tensor]
    labels = node_explanations.label  # current type: list[tensor]
    pred_labels_hard = node_explanations.pred_label_hard  # current type: list[tensor]
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
    pred_labels_hard = node_explanations.pred_label_hard  # current type: list[tensor]
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
    score = (pos_weight + neg_weight) / (
        pos_weight / fidelity_pos_score + neg_weight / (1 - fidelity_neg_score))
    return score


def fidelity_curve_auc(node_explanations):
    """
    calculate the AUC of the fidelity curve, based on the implementation in `torch_geometric.explain.metric.fidelity_curve_auc`
    introduced in <https://arxiv.org/abs/2206.09677>
    :param node_explanations: a NodeExplanationCombination object which contains a list of NodeExplanation objects
    NodeExplanationCombination object has the following key, value in control_data:
    auc_threshold_values: the threshold values to threshold the soft mask, like [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    NodeExplanation object has the following attributes:
    label: ground truth label of the node
    masked_pred_label_hard_threshold:
        a list of predicted labels of the node after masking while soft mask is thresholded by different values
        threshold values should be a list of threshold values to threshold the soft mask,
        Example:
        masked_pred_label_hard_threshold = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        # auc_threshold_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    opposite_masked_pred_label_hard_threshold:
        a list of predicted labels of the node after masking by the opposite mask while soft mask is thresholded by different values
        threshold values should be a list of threshold values to threshold the soft mask,
        Example:
        opposite_masked_pred_label_hard_threshold = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        # auc_threshold_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    :return:
    """
    auc_threshold_values = node_explanations.control_data.get('auc_threshold_values',
                                                              None)
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

    x = torch.tensor(auc_threshold_values)
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
    if node_explanations.control_data.get('sparsity_type', 'edge') == 'edge':
        edge_masks = node_explanations.edge_mask_hard  # current type: list[list[tensor]]
        sparsity_score = sum([sparsity_core(mask) for mask in edge_masks]) / len(edge_masks)
    else:
        feature_masks = node_explanations.feature_mask_hard
        sparsity_score = sum([sparsity_core(mask) for mask in feature_masks]) / len(feature_masks)
    return sparsity_score


def sparsity_core(edge_mask):
    total = sum(edge_mask[i].sum().item() for i in range(len(edge_mask)))
    return total / sum(edge_mask[i].numel() for i in range(len(edge_mask)))


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
    return max(scores).item()


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
    masked_pred_labels_hard = node_explanations.masked_pred_label  # current type: list[tensor]
    labels = node_explanations.label  # current type: list[tensor]
    score = roc_auc_core(masked_pred_labels_hard, labels)
    return score


def roc_auc_core(masked, label):
    masked = torch.stack(masked).softmax(1).cpu().numpy()
    label = torch.stack(label)
    num_classes = masked.shape[1]
    current_class = len(torch.unique(label))
    label = label.cpu().numpy()
    if current_class < 2:
        print('Only one class present in the data, returning 0')
        return 0
    from sklearn.metrics import roc_auc_score as roc_auc_score_sklearn
    # test if multi-class
    if num_classes > 2:
        return roc_auc_score_sklearn(label, masked, multi_class='ovr',
                                     labels=list(range(num_classes)))
    return roc_auc_score_sklearn(label, masked)


def fidelity_pos_model(node_explanations):
    opposite_masked_pred_label_hard = node_explanations.opposite_masked_pred_label_hard  # current type: list[tensor]
    pred_labels_hard = node_explanations.pred_label_hard  # current type: list[tensor]
    score = fidelity_model_core(opposite_masked_pred_label_hard, pred_labels_hard)
    return 1 - score


def fidelity_neg_model(node_explanations):
    masked_pred_label_hard = node_explanations.masked_pred_label_hard  # current type: list[tensor]
    pred_labels_hard = node_explanations.pred_label_hard  # current type: list[tensor]
    score = fidelity_model_core(masked_pred_label_hard, pred_labels_hard)
    return 1 - score


def fidelity_model_core(masked, pred):
    comparison = [masked[i].eq(pred[i]) for i in range(len(masked))]
    return (sum(comparison) / len(comparison)).item()


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
    'roc_auc_score': roc_auc_score,
    'fidelity_pos_model': fidelity_pos_model,
    'fidelity_neg_model': fidelity_neg_model,
}
