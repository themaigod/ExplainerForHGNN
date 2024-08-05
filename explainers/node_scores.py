import torch


def mask_density(node_explanation):
    """
    explanation mask density, introduced in GNNExplainer
    :param node_explanation: a NodeExplanation object with edge_mask and neighbors_gs
    """

    edge_mask = node_explanation.edge_mask

    if isinstance(edge_mask, torch.Tensor):
        mask_density_score = mask_density_core(edge_mask)

    elif isinstance(edge_mask, list):
        mask_density_score = 0
        for mask in edge_mask:
            mask_density_score += mask_density_core(mask)
        mask_density_score /= len(edge_mask)
    else:
        raise ValueError('edge_mask should be a tensor or a list of tensors')

    return mask_density_score


def mask_density_core(edge_mask):
    """
    :param edge_mask: a tensor of shape (E,)
    :return:
    """
    edge_mask = edge_mask.clone().detach()
    edge_mask[edge_mask < 0] = 0  # remove negative values

    mask_density_score = edge_mask.sum() / edge_mask.size(0)
    return mask_density_score.item()


node_scores = {
    'mask_density': mask_density,
}
