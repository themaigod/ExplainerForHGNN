import torch


def get_edge_mask_hard(explainer, opposite=False):
    if explainer.config['edge_mask_hard_method'] == 'threshold':
        edge_mask = explainer.edge_mask_for_output
        threshold = explainer.config['edge_mask_threshold']
        if opposite:
            return [em <= threshold for em in edge_mask]
        return [em > threshold for em in edge_mask]
    elif explainer.config['edge_mask_hard_method'] == 'auto_threshold':
        edge_mask = explainer.edge_mask_for_output
        edge_mask_concat = torch.cat(edge_mask)
        threshold = torch.quantile(edge_mask_concat,
                                   explainer.config['threshold_percentage_edge'])
        if opposite:
            return [em <= threshold for em in edge_mask]
        return [em > threshold for em in edge_mask]
    elif explainer.config['edge_mask_hard_method'] == 'original':
        if opposite:
            return [-em for em in explainer.edge_mask_for_output]
        return explainer.edge_mask_for_output
    elif explainer.config['edge_mask_hard_method'] == 'top_k':
        edge_mask = explainer.edge_mask_for_output
        top_k = explainer.config['top_k_for_edge_mask']
        return get_top_k_edge_mask_core(edge_mask, top_k, opposite)
    else:
        raise ValueError('Invalid edge_mask_threshold_method: {}'.format(
            explainer.config['edge_mask_threshold_method']))


def get_top_k_edge_mask_core(edge_mask, top_k, opposite=False):
    edge_mask_concat = torch.cat(edge_mask)
    top_k = int(top_k * len(edge_mask_concat))
    indices = torch.sort(edge_mask_concat, descending=True)[1][:top_k]
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

    if not opposite:
        edge_mask_hard = [torch.zeros_like(em) for em in edge_mask]
        for i, j in indices_ori:
            edge_mask_hard[i][j] = 1
        return edge_mask_hard
    else:
        edge_mask_hard = [torch.ones_like(em) for em in edge_mask]
        for i, j in indices_ori:
            edge_mask_hard[i][j] = 0
        return edge_mask_hard


def get_masked_gs_hard(explainer, opposite=False):
    if explainer.config['edge_mask_hard_method'] == 'original':
        if not opposite:
            return explainer.masked['masked_gs']
        else:
            return [-g.coalesce() for g in explainer.masked['opposite_masked_gs']]
    edge_mask_hard = get_edge_mask_hard(explainer, opposite)
    gs = explainer.neighbor_input['gs']
    masked_gs_hard = []
    for g, em in zip(gs, edge_mask_hard):
        g = g.coalesce()
        values = g.values()
        mask = em
        masked_g = torch.sparse_coo_tensor(g.indices()[:, mask], values[mask],
                                           g.size())
        masked_gs_hard.append(masked_g)

    return masked_gs_hard


def get_masked_gs_hard_core(gs, edge_mask_hard):
    masked_gs_hard = []
    for g, em in zip(gs, edge_mask_hard):
        g = g.coalesce()
        values = g.values()
        mask = em
        masked_g = torch.sparse_coo_tensor(g.indices()[:, mask], values[mask],
                                           g.size())
        masked_gs_hard.append(masked_g)

    return masked_gs_hard


def get_feature_mask_hard(explainer, opposite=False):
    if explainer.feature_mask_for_output is None:
        return None

    if explainer.config['feature_mask_hard_method'] == 'threshold':
        feature_mask = explainer.feature_mask_for_output
        threshold = explainer.config['feature_mask_threshold']
        if opposite:
            return [fm <= threshold for fm in feature_mask]
        return feature_mask > threshold
    elif explainer.config['feature_mask_hard_method'] == 'auto_threshold':
        feature_mask = explainer.feature_mask_for_output
        threshold = torch.quantile(feature_mask,
                                   explainer.config['threshold_percentage_feature'])
        if opposite:
            return feature_mask <= threshold
        return feature_mask > threshold
    elif explainer.config['feature_mask_hard_method'] == 'original':
        return explainer.feature_mask_for_output
    elif explainer.config['feature_mask_hard_method'] == 'top_k':
        feature_mask = explainer.feature_mask_for_output
        top_k = int(explainer.config['top_k_for_feature_mask'] * len(feature_mask))
        indices = torch.sort(feature_mask, descending=True)[1][top_k]
        if opposite:
            feature_mask_hard = torch.ones_like(feature_mask)
            feature_mask_hard[indices] = 0
        else:
            feature_mask_hard = torch.zeros_like(feature_mask)
            feature_mask_hard[indices] = 1
        return feature_mask_hard
    else:
        raise ValueError('Invalid feature_mask_threshold_method: {}'.format(
            explainer.config['feature_mask_threshold_method']))


def fidelity_neg_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label_hard, pred_label_hard and label for the fidelity- score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label_hard" not in node_explanation:
        if "masked_pred_label" not in node_explanation:
            if "masked_gs_hard" not in node_explanation:
                masked_gs_hard = get_masked_gs_hard(explainer)
                node_explanation.masked_gs_hard = masked_gs_hard

            if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                       'feature_mask_for_output',
                                                                       None) is not None:
                feature_mask_hard = get_feature_mask_hard(explainer)
                node_explanation.feature_mask_hard = feature_mask_hard

            masked_pred_label = \
                explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
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

    if "label" not in node_explanation:
        label_test = explainer.model.dataset.labels[2]
        label = None
        for l in label_test:
            if l[0] == explainer.node_id:
                label = l[1]
                break

        label = torch.tensor(label, device=explainer.model.device)
        node_explanation.label = label

    return node_explanation


def fidelity_pos_explanation(node_explanation, explainer):
    """
    Set the opposite_masked_pred_label_hard, opposite_masked_pred_label and label for the fidelity+ score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "opposite_masked_pred_label_hard" not in node_explanation:
        if "opposite_masked_pred_label" not in node_explanation:
            if "opposite_masked_gs_hard" not in node_explanation:
                opposite_masked_gs_hard = get_masked_gs_hard(explainer, opposite=True)
                node_explanation.opposite_masked_gs_hard = opposite_masked_gs_hard

            if "opposite_feature_mask_hard" not in node_explanation and getattr(
                explainer, 'feature_mask_for_output', None) is not None:
                opposite_feature_mask_hard = get_feature_mask_hard(explainer,
                                                                   opposite=True)
                node_explanation.opposite_feature_mask_hard = opposite_feature_mask_hard

            opposite_masked_pred_label = \
                explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
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

    if "label" not in node_explanation:
        label_test = explainer.model.dataset.labels[2]
        label = None
        for l in label_test:
            if l[0] == explainer.node_id:
                label = l[1]
                break

        label = torch.tensor(label, device=explainer.model.device)
        node_explanation.label = label

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


def get_multi_edge_threshold_auc(explainer):
    if explainer.config['edge_mask_hard_method'] == 'threshold':
        if explainer.config.get(['threshold_auc'], None) is not None:
            return explainer.config['threshold_auc']
        else:
            num = 11
            return [i / (num - 1) for i in range(num)]
    elif explainer.config['edge_mask_hard_method'] == 'auto_threshold':
        edge_mask = explainer.edge_mask_for_output
        edge_mask_concat = torch.cat(edge_mask)
        threshold = [torch.quantile(edge_mask_concat, i) for i in
                     explainer.config['threshold_percentage_auc']]
        return threshold
    elif explainer.config['edge_mask_hard_method'] == 'original':
        if explainer.config.get(['threshold_auc'], None) is not None:
            return explainer.config['threshold_auc']
        else:
            num = 11
            return [i / (num - 1) for i in range(num)]
    elif explainer.config['edge_mask_hard_method'] == 'top_k':
        if explainer.config.get(['top_k_for_auc'], None) is not None:
            return explainer.config['top_k_for_auc']
        else:
            num = 11
            return [i / (num - 1) for i in range(num)]
    else:
        raise ValueError('Invalid edge_mask_threshold_method: {}'.format(
            explainer.config['edge_mask_threshold_method']))


def get_edge_mask_hard_threshold_auc(explainer, opposite=False):
    threshold_values = get_multi_edge_threshold_auc(explainer)
    edge_mask = explainer.edge_mask_for_output
    if explainer.config['edge_mask_hard_method'] == 'top_k':
        edge_masks_hard = []
        for threshold in threshold_values:
            edge_mask_hard = get_top_k_edge_mask_core(edge_mask, threshold, opposite)
            edge_masks_hard.append(edge_mask_hard)
        return edge_masks_hard
    else:
        if not opposite:
            return [[em > threshold for em in edge_mask] for threshold in
                    threshold_values]
        else:
            return [[em <= threshold for em in edge_mask] for threshold in
                    threshold_values]


def get_masked_gs_hard_threshold_auc(explainer, opposite=False):
    gs = explainer.neighbor_input['gs']
    masked_gs_threshold = []
    edge_masks = get_edge_mask_hard_threshold_auc(explainer, opposite)
    for edge_mask_hard in edge_masks:
        masked_gs_hard = get_masked_gs_hard_core(gs, edge_mask_hard)
        masked_gs_threshold.append(masked_gs_hard)
    return masked_gs_threshold


def get_top_k_feature_mask_core(feature_mask, top_k, opposite=False):
    top_k = int(top_k * len(feature_mask))
    indices = torch.sort(feature_mask, descending=True)[1][:top_k]
    if opposite:
        feature_mask_hard = torch.ones_like(feature_mask)
        feature_mask_hard[indices] = 0
    else:
        feature_mask_hard = torch.zeros_like(feature_mask)
        feature_mask_hard[indices] = 1
    return feature_mask_hard


def get_multi_feature_threshold_auc(explainer):
    if explainer.config['feature_mask_hard_method'] == 'threshold':
        if explainer.config.get(['threshold_auc'], None) is not None:
            return explainer.config['threshold_auc']
        else:
            num = 11
            return [i / (num - 1) for i in range(num)]
    elif explainer.config['feature_mask_hard_method'] == 'auto_threshold':
        feature_mask = explainer.feature_mask_for_output
        threshold = [torch.quantile(feature_mask, i) for i in
                     explainer.config['threshold_percentage_auc']]
        return threshold
    elif explainer.config['feature_mask_hard_method'] == 'original':
        if explainer.config.get(['threshold_auc'], None) is not None:
            return explainer.config['threshold_auc']
        else:
            num = 11
            return [i / (num - 1) for i in range(num)]
    elif explainer.config['feature_mask_hard_method'] == 'top_k':
        if explainer.config.get(['top_k_for_auc'], None) is not None:
            return explainer.config['top_k_for_auc']
        else:
            num = 11
            return [i / (num - 1) for i in range(num)]
    else:
        raise ValueError('Invalid feature_mask_threshold_method: {}'.format(
            explainer.config['feature_mask_threshold_method']))


def get_feature_mask_hard_threshold_auc(explainer, opposite=False):
    threshold_values = get_multi_feature_threshold_auc(explainer)
    feature_mask = explainer.feature_mask_for_output
    if explainer.config['feature_mask_hard_method'] == 'top_k':
        feature_masks_hard = []
        for threshold in threshold_values:
            feature_mask_hard = get_top_k_feature_mask_core(feature_mask, threshold,
                                                            opposite)
            feature_masks_hard.append(feature_mask_hard)
        return feature_masks_hard
    else:
        if not opposite:
            return [feature_mask > threshold for threshold in threshold_values]
        else:
            return [feature_mask <= threshold for threshold in threshold_values]


def fidelity_curve_auc_explanation(node_explanation, explainer):
    """

    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "label" not in node_explanation:
        label_test = explainer.model.dataset.labels[2]
        label = None
        for l in label_test:
            if l[0] == explainer.node_id:
                label = l[1]
                break

        label = torch.tensor(label, device=explainer.model.device)
        node_explanation.label = label

    if "pred_label_hard" not in node_explanation:
        if "pred_label" not in node_explanation:
            pred_label = explainer.model()[explainer.node_id]
            node_explanation.pred_label = pred_label

        pred_label_hard = node_explanation.pred_label.argmax()
        node_explanation.pred_label_hard = pred_label_hard

    if "masked_pred_labels_hard_threshold" not in node_explanation:
        if not explainer.config.get('auc_use_feature_mask', False) and \
            not explainer.config.get('auc_use_edge_mask', False):
            raise ValueError(
                'one of auc_use_feature_mask and auc_use_edge_mask should be True')
        if explainer.config.get('auc_use_feature_mask', False):
            feature_masks_hard = get_feature_mask_hard_threshold_auc(explainer)
        else:
            feature_masks_hard = None
        if explainer.config.get('auc_use_edge_mask', False):
            masked_gs = get_masked_gs_hard_threshold_auc(explainer)
        else:
            masked_gs = None
        masked_pred_labels_hard_threshold = []
        length = len(feature_masks_hard) if feature_masks_hard is not None else len(
            masked_gs)
        for i in range(length):
            masked_pred_label = \
                explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                    masked_gs[i], feature_masks_hard[i]))[
                    explainer.node_id]
            masked_pred_label_hard = masked_pred_label.argmax()
            masked_pred_labels_hard_threshold.append(masked_pred_label_hard)

        node_explanation.masked_pred_labels_hard_threshold = masked_pred_labels_hard_threshold

    if "opposite_masked_pred_labels_hard_threshold" not in node_explanation:
        if not explainer.config.get('auc_use_feature_mask', False) and \
            not explainer.config.get('auc_use_edge_mask', False):
            raise ValueError(
                'one of auc_use_feature_mask and auc_use_edge_mask should be True')
        if explainer.config.get('auc_use_feature_mask', False):
            feature_masks_hard = get_feature_mask_hard_threshold_auc(explainer,
                                                                     opposite=True)
        else:
            feature_masks_hard = None
        if explainer.config.get('auc_use_edge_mask', False):
            masked_gs = get_masked_gs_hard_threshold_auc(explainer, opposite=True)
        else:
            masked_gs = None
        masked_pred_labels_hard_threshold = []
        length = len(feature_masks_hard) if feature_masks_hard is not None else len(
            masked_gs)
        for i in range(length):
            masked_pred_label = \
                explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                    masked_gs[i], feature_masks_hard[i]))[
                    explainer.node_id]
            masked_pred_label_hard = masked_pred_label.argmax()
            masked_pred_labels_hard_threshold.append(masked_pred_label_hard)

        node_explanation.opposite_masked_pred_labels_hard_threshold = masked_pred_labels_hard_threshold

    return node_explanation


def unfaithfulness_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label and pred_label for the unfaithfulness score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label" not in node_explanation:
        if "masked_gs_hard" not in node_explanation:
            masked_gs_hard = get_masked_gs_hard(explainer)
            node_explanation.masked_gs_hard = masked_gs_hard
        if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                   'feature_mask_for_output',
                                                                   None) is not None:
            feature_mask_hard = get_feature_mask_hard(explainer)
            node_explanation.feature_mask_hard = feature_mask_hard
        masked_pred_label = \
            explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
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
    Set the edge_mask_hard for the sparsity score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "edge_mask_hard" not in node_explanation:
        edge_mask_hard = get_edge_mask_hard(explainer)
        node_explanation.edge_mask_hard = edge_mask_hard

    return node_explanation


def generate_perturbed_feature_mask(feature_mask, k):
    top_k = int(k * len(feature_mask))
    top_k_indices = torch.argsort(feature_mask, descending=True)[top_k:]
    perturbed_feature_mask = feature_mask.clone()
    perturbed_feature_mask[top_k_indices] = torch.normal(0, 0.1,
                                                         size=(len(top_k_indices),),
                                                         device=feature_mask.device)
    return perturbed_feature_mask


def graph_exp_faith_feature_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label and perturbed_feature_pred_label for the faithfulness score

    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label" not in node_explanation:
        if "masked_gs_hard" not in node_explanation:
            masked_gs_hard = get_masked_gs_hard(explainer)
            node_explanation.masked_gs_hard = masked_gs_hard
        if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                   'feature_mask_for_output',
                                                                   None) is not None:
            feature_mask_hard = get_feature_mask_hard(explainer)
            node_explanation.feature_mask_hard = feature_mask_hard
        masked_pred_label = \
            explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.feature_mask_hard))[
                explainer.node_id]
        node_explanation.masked_pred_label = masked_pred_label

    if "perturbed_feature_pred_label" not in node_explanation:
        if "perturbed_feature" not in node_explanation:
            perturbed_feature = generate_perturbed_feature_mask(
                explainer.feature_mask_for_output,
                explainer.config['top_k_for_faith_feature'])
            node_explanation.perturbed_feature = perturbed_feature
        perturbed_feature_pred_label = \
            explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.perturbed_feature))[
                explainer.node_id]
        node_explanation.perturbed_feature_pred_label = perturbed_feature_pred_label

    return node_explanation


def generate_perturbed_edge_mask(edge_mask, k):
    edge_mask = edge_mask[0]
    top_k = int(k * len(edge_mask))
    top_k_indices = torch.argsort(edge_mask, descending=True)[top_k:]
    perturbed_edge_mask = torch.zeros_like(edge_mask)
    perturbed_edge_mask[top_k_indices] = 1
    return perturbed_edge_mask


def graph_exp_faith_edge_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label and perturbed_edge_pred_label for the faithfulness score

    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label" not in node_explanation:
        if "masked_gs_hard" not in node_explanation:
            masked_gs_hard = get_masked_gs_hard(explainer)
            node_explanation.masked_gs_hard = masked_gs_hard
        if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                   'feature_mask_for_output',
                                                                   None) is not None:
            feature_mask_hard = get_feature_mask_hard(explainer)
            node_explanation.feature_mask_hard = feature_mask_hard
        masked_pred_label = \
            explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.feature_mask_hard))[
                explainer.node_id]
        node_explanation.masked_pred_label = masked_pred_label

    if "perturbed_edge_pred_label" not in node_explanation:
        if "perturbed_edge" not in node_explanation:
            perturbed_edge = generate_perturbed_edge_mask(
                explainer.edge_mask_for_output,
                explainer.config['top_k_for_faith_edge'])
            node_explanation.perturbed_edge = perturbed_edge
        perturbed_edge_pred_label = \
            explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.feature_mask_hard))[
                explainer.node_id]
        node_explanation.perturbed_edge_pred_label = perturbed_edge_pred_label

    return node_explanation


def generate_perturbed_gs(gs, perturb_ratio=0.001, neighbors_all=None):
    new_gs = []
    if neighbors_all is None:
        neighbors_all = []
        for g in gs:
            g = g.coalesce()
            edge_index = g.indices().clone()

            neighbors = {}
            for i in range(edge_index.size(1)):
                if edge_index[0, i].item() not in neighbors:
                    neighbors[edge_index[0, i].item()] = set()
                neighbors[edge_index[0, i].item()].add(edge_index[1, i].item())
            neighbors_all.append(neighbors)

    for idx, g in enumerate(gs):
        g = g.coalesce()
        edge_index = g.indices().clone()

        neighbors = neighbors_all[idx]

        def check_edge(i, j):
            return i in neighbors and j in neighbors[i]

        swap_times = round(edge_index.size(1) * perturb_ratio)
        swap_count = 0
        while swap_count < swap_times:
            i = torch.randint(0, edge_index.size(1), (1,)).item()
            j = torch.randint(0, edge_index.size(1), (1,)).item()
            if i == j:  # avoid the same index
                continue

            # avoid the same target node
            if edge_index[0, i] == edge_index[0, j] or edge_index[1, i] == edge_index[
                1, j]:
                continue

            # avoid existing edge
            # (edge_index[0, i], edge_index[1, j]) or (edge_index[0, j], edge_index[1, i]) is an existing edge
            if check_edge(edge_index[0, i].item(),
                          edge_index[1, j].item()) or check_edge(
                edge_index[0, j].item(), edge_index[1, i].item()):
                continue

            edge_index[0, i], edge_index[0, j] = edge_index[0, j], edge_index[0, i]
            swap_count += 1

        new_g = torch.sparse_coo_tensor(edge_index, g.values(), g.size())
        new_g = new_g.coalesce()
        new_gs.append(new_g)
    return new_gs, neighbors_all


def generate_perturbed_features(features, node_id, std=0.01):
    features = features.clone().detach()
    features[node_id] += torch.normal(0, std, size=(features.size(1),),
                                      device=features.device)
    return features


def generate_perturbed_mask_for_stability(explainer, neighbors=None):
    import copy
    config = copy.deepcopy(explainer.config)
    config["eval_metrics"] = []
    config["summary_path"] = None
    config["record_step"] = config["epochs"] - 1
    new_explainer = explainer.__class__(config)
    new_model = copy.deepcopy(explainer.model)
    gs, features = new_model.standard_input()
    pert_gs, neighbors = generate_perturbed_gs(gs, explainer.config.get(
        'perturb_ratio_in_gs_for_stability', 0.001), neighbors)
    pert_features = generate_perturbed_features(features, explainer.node_id,
                                                std=explainer.config.get(
                                                    'perturb_std_in_features_for_stability',
                                                    0.01))

    def standard_input(self):
        return pert_gs, pert_features

    import types
    new_model.standard_input = types.MethodType(standard_input, new_model)
    new_explainer.explain(new_model, node_id=explainer.node_id)
    if getattr(new_explainer, 'feature_mask_for_output', None) is not None:
        perturbed_feature_mask = new_explainer.feature_mask_for_output
    else:
        perturbed_feature_mask = None
    perturbed_edge_mask = new_explainer.edge_mask_for_output

    return perturbed_feature_mask, perturbed_edge_mask, neighbors


def graph_exp_stability_feature_explanation(node_explanation, explainer):
    """
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "feature_mask" not in node_explanation:
        feature_mask = explainer.feature_mask_for_output
        node_explanation.feature_mask = feature_mask

    if "perturbed_feature_mask_for_stability" not in node_explanation:
        perturbed_feature_masks = []
        perturbed_edge_masks = []
        neighbors = None
        for time in range(explainer.config['stability_times']):
            perturbed_feature_mask, perturbed_edge_mask, neighbors = generate_perturbed_mask_for_stability(
                explainer, neighbors)
            perturbed_feature_masks.append(perturbed_feature_mask)
            perturbed_edge_masks.append(perturbed_edge_mask)
        node_explanation.perturbed_feature_mask_for_stability = perturbed_feature_masks
        node_explanation.perturbed_edge_mask_for_stability = perturbed_edge_masks

    return node_explanation


def graph_exp_stability_edge_explanation(node_explanation, explainer):
    """
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "edge_mask" not in node_explanation:
        edge_mask = explainer.edge_mask_for_output
        node_explanation.edge_mask = edge_mask

    if "perturbed_edge_mask_for_stability" not in node_explanation:
        perturbed_feature_masks = []
        perturbed_edge_masks = []
        neighbors = None
        for time in range(explainer.config['stability_times']):
            perturbed_feature_mask, perturbed_edge_mask, neighbors = generate_perturbed_mask_for_stability(
                explainer, neighbors)
            perturbed_feature_masks.append(perturbed_feature_mask)
            perturbed_edge_masks.append(perturbed_edge_mask)
        node_explanation.perturbed_feature_mask_for_stability = perturbed_feature_masks
        node_explanation.perturbed_edge_mask_for_stability = perturbed_edge_masks

    return node_explanation


def MacroF1_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label and pred_label for the MacroF1 score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label" not in node_explanation:
        if "masked_gs_hard" not in node_explanation:
            masked_gs_hard = get_masked_gs_hard(explainer)
            node_explanation.masked_gs_hard = masked_gs_hard
        if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                   'feature_mask_for_output',
                                                                   None) is not None:
            feature_mask_hard = get_feature_mask_hard(explainer)
            node_explanation.feature_mask_hard = feature_mask_hard
        masked_pred_label = \
            explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.feature_mask_hard))[
                explainer.node_id]
        node_explanation.masked_pred_label = masked_pred_label

    if "pred_label" not in node_explanation:
        pred_label = explainer.model()[explainer.node_id]
        node_explanation.pred_label = pred_label

    return node_explanation


def MicroF1_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label and pred_label for the MicroF1 score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label" not in node_explanation:
        if "masked_gs_hard" not in node_explanation:
            masked_gs_hard = get_masked_gs_hard(explainer)
            node_explanation.masked_gs_hard = masked_gs_hard
        if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                   'feature_mask_for_output',
                                                                   None) is not None:
            feature_mask_hard = get_feature_mask_hard(explainer)
            node_explanation.feature_mask_hard = feature_mask_hard
        masked_pred_label = \
            explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.feature_mask_hard))[
                explainer.node_id]
        node_explanation.masked_pred_label = masked_pred_label

    if "pred_label" not in node_explanation:
        pred_label = explainer.model()[explainer.node_id]
        node_explanation.pred_label = pred_label

    return node_explanation


def roc_auc_score_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label and pred_label for the ROC-AUC score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label" not in node_explanation:
        if "masked_gs_hard" not in node_explanation:
            masked_gs_hard = get_masked_gs_hard(explainer)
            node_explanation.masked_gs_hard = masked_gs_hard
        if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                   'feature_mask_for_output',
                                                                   None) is not None:
            feature_mask_hard = get_feature_mask_hard(explainer)
            node_explanation.feature_mask_hard = feature_mask_hard
        masked_pred_label = \
            explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.feature_mask_hard))[
                explainer.node_id]
        node_explanation.masked_pred_label = masked_pred_label

    if "pred_label" not in node_explanation:
        pred_label = explainer.model()[explainer.node_id]
        node_explanation.pred_label = pred_label

    return node_explanation


prepare_explanation_fn_for_node_dataset_scores = {
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
    'Micro-F1': MicroF1_explanation,
    'roc_auc_score': roc_auc_score_explanation
}
