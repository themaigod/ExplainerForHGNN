def mask_density_explanation(node_explanation, explainer):
    """
    Set the edge_mask for the mask_density score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """

    if "edge_mask" not in node_explanation:
        edge_mask = explainer.edge_mask_for_output
        node_explanation.edge_mask = edge_mask

    return node_explanation


prepare_explanation_fn_for_node_scores = {
    'mask_density': mask_density_explanation,
}


def standard_explanation(node_explanation, explainer):
    """
    standard explanation, introduced in GNNExplainer
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "edge_mask" not in node_explanation and getattr(explainer,
                                                       'edge_mask_for_output',
                                                       None) is not None:
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
