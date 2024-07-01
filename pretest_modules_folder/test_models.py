import pytest
import torch
import torch.nn.functional as F

from models.han import GATConv as GATLayer

# @pytest.mark.parametrize("residual", [True, False])
# @pytest.mark.parametrize("bias", [True, False])
# def test_gat_layer_homogeneous_no_edge_weights(residual, bias):
#     in_feats = 5
#     out_feats = 3
#     num_heads = 2
#     num_nodes = 4
#
#     h = torch.rand((num_nodes, in_feats))
#     g = torch.tensor([[1, 1, 0, 0],
#                       [1, 1, 1, 0],
#                       [0, 1, 1, 1],
#                       [0, 0, 1, 1]], dtype=torch.float32)
#     g = g.to_sparse_coo()
#
#     gat_layer = GATLayer(in_feats, out_feats, num_heads, residual=residual, bias=bias)
#     output = gat_layer(h, g)
#
#     assert output.shape == (num_nodes, out_feats * num_heads)
#
#
# @pytest.mark.parametrize("residual", [True, False])
# @pytest.mark.parametrize("bias", [True, False])
# def test_gat_layer_homogeneous_with_edge_weights(residual, bias):
#     in_feats = 5
#     out_feats = 3
#     num_heads = 2
#     num_nodes = 4
#
#     h = torch.rand((num_nodes, in_feats))
#     g = torch.tensor([[1, 1, 0, 0],
#                       [1, 1, 1, 0],
#                       [0, 1, 1, 1],
#                       [0, 0, 1, 1]], dtype=torch.float32)
#     g = g.to_sparse_coo()
#     edge_weights = torch.rand((g._nnz(),))
#
#     gat_layer = GATLayer(in_feats, out_feats, num_heads, residual=residual, bias=bias)
#     output = gat_layer(h, g, edge_weight=edge_weights)
#
#     assert output.shape == (num_nodes, out_feats * num_heads)
#
#
# @pytest.mark.parametrize("residual", [True, False])
# @pytest.mark.parametrize("bias", [True, False])
# def test_gat_layer_bipartite_no_edge_weights(residual, bias):
#     in_feats_src = 5
#     in_feats_dst = 6
#     out_feats = 3
#     num_heads = 2
#     num_src_nodes = 4
#     num_dst_nodes = 3
#
#     h_src = torch.rand((num_src_nodes, in_feats_src))
#     h_dst = torch.rand((num_dst_nodes, in_feats_dst))
#     g = torch.tensor([[1, 1, 0],
#                       [1, 0, 1],
#                       [0, 1, 1],
#                       [0, 0, 1]], dtype=torch.float32)
#     g = g.to_sparse_coo()
#
#     gat_layer = GATLayer((in_feats_src, in_feats_dst), out_feats, num_heads,
#                          residual=residual, bias=bias)
#     output = gat_layer((h_src, h_dst), g)
#
#     assert output.shape == (num_dst_nodes, out_feats * num_heads)
#
#
# @pytest.mark.parametrize("residual", [True, False])
# @pytest.mark.parametrize("bias", [True, False])
# def test_gat_layer_bipartite_with_edge_weights(residual, bias):
#     in_feats_src = 5
#     in_feats_dst = 6
#     out_feats = 3
#     num_heads = 2
#     num_src_nodes = 4
#     num_dst_nodes = 3
#
#     h_src = torch.rand((num_src_nodes, in_feats_src))
#     h_dst = torch.rand((num_dst_nodes, in_feats_dst))
#     g = torch.tensor([[1, 1, 0],
#                       [1, 0, 1],
#                       [0, 1, 1],
#                       [0, 0, 1]], dtype=torch.float32)
#     g = g.to_sparse_coo()
#     edge_weights = torch.rand((g._nnz(),))
#
#     gat_layer = GATLayer((in_feats_src, in_feats_dst), out_feats, num_heads,
#                          residual=residual, bias=bias)
#     output = gat_layer((h_src, h_dst), g, edge_weight=edge_weights)
#
#     assert output.shape == (num_dst_nodes, out_feats * num_heads)
#
#
# @pytest.mark.parametrize("residual", [True, False])
# @pytest.mark.parametrize("bias", [True, False])
# def test_gat_layer_zero_in_degree(residual, bias):
#     in_feats = 5
#     out_feats = 3
#     num_heads = 2
#     num_nodes = 4
#
#     h = torch.rand((num_nodes, in_feats))
#     g = torch.tensor([[0, 0, 0, 0],
#                       [1, 0, 0, 0],
#                       [0, 1, 0, 0],
#                       [0, 0, 1, 0]], dtype=torch.float32)
#     g = g.to_sparse_coo()
#
#     gat_layer = GATLayer(in_feats, out_feats, num_heads, residual=residual, bias=bias,
#                          allow_zero_in_degree=True)
#     output = gat_layer(h, g)
#
#     assert output.shape == (num_nodes, out_feats * num_heads)


import dgl
from dgl.nn.pytorch import GATConv


# Import your GATLayer implementation here
# from your_module import GATLayer

def copy_params(src_layer, dst_layer):
    # Copy weights
    for name, param in src_layer.named_parameters():
        getattr(dst_layer, name).data = param.data.detach().clone()


def compare_gradients(tensor1, tensor2):
    return torch.allclose(tensor1.grad, tensor2.grad, atol=1e-6)


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_gat_layer_homogeneous_no_edge_weights(residual, bias):
    in_feats = 5
    out_feats = 3
    num_heads = 2
    num_nodes = 4

    h = torch.rand((num_nodes, in_feats), requires_grad=True)
    adj_matrix = torch.tensor([[1, 1, 0, 0],
                               [1, 1, 1, 0],
                               [0, 1, 1, 1],
                               [0, 0, 1, 1]], dtype=torch.float32)

    # Convert adjacency matrix to DGL graph
    g = dgl.graph((adj_matrix.nonzero(as_tuple=True)))

    adj_matrix = adj_matrix.to_sparse_coo()

    # Your GATLayer
    gat_layer = GATLayer(in_feats, out_feats, num_heads, residual=residual, bias=bias)
    output_custom = gat_layer(h, adj_matrix)

    # DGL GATConv
    gat_conv = GATConv(in_feats, out_feats, num_heads, feat_drop=0.0, attn_drop=0.0,
                       negative_slope=0.2, residual=residual, activation=None)
    copy_params(gat_layer, gat_conv)
    output_dgl = gat_conv(g, h)

    assert output_custom.shape == (num_nodes, out_feats * num_heads)
    assert torch.allclose(output_custom, output_dgl.view(num_nodes, -1), atol=1e-6)

    # Compare gradients
    output_custom.sum().backward(retain_graph=True)
    output_dgl.sum().backward()
    assert compare_gradients(h, h)


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_gat_layer_homogeneous_with_edge_weights(residual, bias):
    in_feats = 5
    out_feats = 3
    num_heads = 2
    num_nodes = 4

    h = torch.rand((num_nodes, in_feats), requires_grad=True)
    adj_matrix = torch.tensor([[1, 1, 0, 0],
                               [1, 1, 1, 0],
                               [0, 1, 1, 1],
                               [0, 0, 1, 1]], dtype=torch.float32)

    # Convert adjacency matrix to DGL graph
    g = dgl.graph((adj_matrix.nonzero(as_tuple=True)))
    edge_weights = torch.rand((g.num_edges(),), requires_grad=True)

    adj_matrix = adj_matrix.to_sparse_coo()

    # Your GATLayer
    gat_layer = GATLayer(in_feats, out_feats, num_heads, residual=residual, bias=bias)
    output_custom = gat_layer(h, adj_matrix, edge_weight=edge_weights)

    # DGL GATConv
    gat_conv = GATConv(in_feats, out_feats, num_heads, feat_drop=0.0, attn_drop=0.0,
                       negative_slope=0.2, residual=residual, activation=None)
    copy_params(gat_layer, gat_conv)
    output_dgl = gat_conv(g, h, edge_weight=edge_weights)

    assert output_custom.shape == (num_nodes, out_feats * num_heads)
    assert torch.allclose(output_custom, output_dgl.view(num_nodes, -1), atol=1e-6)

    # Compare gradients
    output_custom.sum().backward(retain_graph=True)
    output_dgl.sum().backward()
    assert compare_gradients(h, h)
    assert compare_gradients(edge_weights, edge_weights)


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_gat_layer_bipartite_no_edge_weights(residual, bias):
    in_feats_src = 5
    in_feats_dst = 6
    out_feats = 3
    num_heads = 2
    num_src_nodes = 4
    num_dst_nodes = 3

    h_src = torch.rand((num_src_nodes, in_feats_src), requires_grad=True)
    h_dst = torch.rand((num_dst_nodes, in_feats_dst), requires_grad=True)
    adj_matrix = torch.tensor([[1, 1, 0],
                               [1, 0, 1],
                               [0, 1, 1],
                               [0, 0, 1]], dtype=torch.float32)

    # Convert adjacency matrix to DGL bipartite graph
    src, dst = adj_matrix.nonzero(as_tuple=True)
    g = dgl.heterograph({('src', 'edge', 'dst'): (src, dst)})

    adj_matrix = adj_matrix.to_sparse_coo()

    # Your GATLayer
    gat_layer = GATLayer((in_feats_src, in_feats_dst), out_feats, num_heads,
                         residual=residual, bias=bias)
    output_custom = gat_layer((h_src, h_dst), adj_matrix)

    # DGL GATConv
    gat_conv = GATConv((in_feats_src, in_feats_dst), out_feats, num_heads,
                       feat_drop=0.0, attn_drop=0.0, negative_slope=0.2,
                       residual=residual, activation=None)
    copy_params(gat_layer, gat_conv)
    output_dgl = gat_conv(g, (h_src, h_dst))

    assert output_custom.shape == (num_dst_nodes, out_feats * num_heads)
    assert torch.allclose(output_custom, output_dgl.view(num_dst_nodes, -1), atol=1e-6)

    # Compare gradients
    output_custom.sum().backward(retain_graph=True)
    output_dgl.sum().backward()
    assert compare_gradients(h_src, h_src)
    assert compare_gradients(h_dst, h_dst)


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_gat_layer_bipartite_with_edge_weights(residual, bias):
    in_feats_src = 5
    in_feats_dst = 6
    out_feats = 3
    num_heads = 2
    num_src_nodes = 4
    num_dst_nodes = 3

    h_src = torch.rand((num_src_nodes, in_feats_src), requires_grad=True)
    h_dst = torch.rand((num_dst_nodes, in_feats_dst), requires_grad=True)
    adj_matrix = torch.tensor([[1, 1, 0],
                               [1, 0, 1],
                               [0, 1, 1],
                               [0, 0, 1]], dtype=torch.float32)

    # Convert adjacency matrix to DGL bipartite graph
    src, dst = adj_matrix.nonzero(as_tuple=True)
    g = dgl.heterograph({('src', 'edge', 'dst'): (src, dst)})
    edge_weights = torch.rand((g.num_edges(),), requires_grad=True)

    adj_matrix = adj_matrix.to_sparse_coo()

    # Your GATLayer
    gat_layer = GATLayer((in_feats_src, in_feats_dst), out_feats, num_heads,
                         residual=residual, bias=bias)
    output_custom = gat_layer((h_src, h_dst), adj_matrix, edge_weight=edge_weights)

    # DGL GATConv
    gat_conv = GATConv((in_feats_src, in_feats_dst), out_feats, num_heads,
                       feat_drop=0.0, attn_drop=0.0, negative_slope=0.2,
                       residual=residual, activation=None)
    copy_params(gat_layer, gat_conv)
    output_dgl = gat_conv(g, (h_src, h_dst), edge_weight=edge_weights)

    assert output_custom.shape == (num_dst_nodes, out_feats * num_heads)
    assert torch.allclose(output_custom, output_dgl.view(num_dst_nodes, -1), atol=1e-6)

    # Compare gradients
    output_custom.sum().backward(retain_graph=True)
    output_dgl.sum().backward()
    assert compare_gradients(h_src, h_src)
    assert compare_gradients(h_dst, h_dst)
    assert compare_gradients(edge_weights, edge_weights)


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_gat_layer_zero_in_degree(residual, bias):
    in_feats = 5
    out_feats = 3
    num_heads = 2
    num_nodes = 4

    h = torch.rand((num_nodes, in_feats), requires_grad=True)
    adj_matrix = torch.tensor([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0]], dtype=torch.float32)

    # Convert adjacency matrix to DGL graph
    g = dgl.graph((adj_matrix.nonzero(as_tuple=True)))

    adj_matrix = adj_matrix.to_sparse_coo()

    # Your GATLayer
    gat_layer = GATLayer(in_feats, out_feats, num_heads, residual=residual, bias=bias,
                         allow_zero_in_degree=True)
    output_custom = gat_layer(h, adj_matrix)

    # DGL GATConv
    gat_conv = GATConv(in_feats, out_feats, num_heads, feat_drop=0.0, attn_drop=0.0,
                       negative_slope=0.2, residual=residual, activation=None,
                       allow_zero_in_degree=True)
    copy_params(gat_layer, gat_conv)
    output_dgl = gat_conv(g, h)

    assert output_custom.shape == (num_nodes, out_feats * num_heads)
    assert torch.allclose(output_custom, output_dgl.view(num_nodes, -1), atol=1e-6)

    # Compare gradients
    output_custom.sum().backward(retain_graph=True)
    output_dgl.sum().backward()
    assert compare_gradients(h, h)


def pretest_gat_layer():
    pytest.main([__file__])
