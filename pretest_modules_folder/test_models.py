import pytest
import torch
import torch.nn.functional as F

from models.han import GATConv as GATLayer
from models.han import HAN
from datasets import ACM

import dgl
from dgl.nn.pytorch import GATConv


def copy_params(src_layer, dst_layer):
    # Copy weights
    dst_name_params = dict(dst_layer.named_parameters())
    for name, param in src_layer.named_parameters():
        dst_name_params[name].data.copy_(param.data.detach().clone())


def compare_gradients(src_layer, dst_layer):
    # Compare gradients
    src_name_params = dict(src_layer.named_parameters())
    dst_name_params = dict(dst_layer.named_parameters())
    for name, param in src_name_params.items():
        if not torch.allclose(param.grad, dst_name_params[name].grad, atol=1e-6):
            return False
    return True


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

    # Convert adjacency matrix to torch sparse tensor
    adj_matrix = adj_matrix.to_sparse_coo()

    # Custom GATLayer
    gat_layer = GATLayer(in_feats, out_feats, num_heads, residual=residual, bias=bias)
    output_custom = gat_layer(h, adj_matrix)

    # DGL GATConv
    gat_conv = GATConv(in_feats, out_feats, num_heads, feat_drop=0.0, attn_drop=0.0,
                       negative_slope=0.2, residual=residual, activation=None)
    copy_params(gat_layer, gat_conv)
    output_dgl = gat_conv(g, h)

    assert output_custom.shape == (num_nodes, num_heads, out_feats)
    assert torch.allclose(output_custom, output_dgl, atol=1e-6)

    # Compare gradients
    output_custom.sum().backward(retain_graph=True)
    output_dgl.sum().backward()
    assert compare_gradients(gat_layer, gat_conv)


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

    # Convert adjacency matrix to torch sparse tensor
    adj_matrix = adj_matrix.to_sparse_coo()

    # Custom GATLayer
    gat_layer = GATLayer(in_feats, out_feats, num_heads, residual=residual, bias=bias)
    output_custom = gat_layer(h, adj_matrix, edge_weight=edge_weights)

    # DGL GATConv
    gat_conv = GATConv(in_feats, out_feats, num_heads, feat_drop=0.0, attn_drop=0.0,
                       negative_slope=0.2, residual=residual, activation=None,
                       bias=bias)
    copy_params(gat_layer, gat_conv)
    output_dgl = gat_conv(g, h, edge_weight=edge_weights)

    assert output_custom.shape == (num_nodes, num_heads, out_feats)
    assert torch.allclose(output_custom, output_dgl, atol=1e-6)

    # Compare gradients
    output_custom.sum().backward(retain_graph=True)
    output_dgl.sum().backward()
    assert compare_gradients(gat_layer, gat_conv)


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

    # Convert adjacency matrix to torch sparse tensor
    adj_matrix = adj_matrix.to_sparse_coo()

    # Custom GATLayer
    gat_layer = GATLayer((in_feats_src, in_feats_dst), out_feats, num_heads,
                         residual=residual, bias=bias)
    output_custom = gat_layer((h_src, h_dst), adj_matrix)

    # DGL GATConv
    gat_conv = GATConv((in_feats_src, in_feats_dst), out_feats, num_heads,
                       feat_drop=0.0, attn_drop=0.0, negative_slope=0.2,
                       residual=residual, activation=None, bias=bias)
    copy_params(gat_layer, gat_conv)
    output_dgl = gat_conv(g, (h_src, h_dst))

    assert output_custom.shape == (num_dst_nodes, num_heads, out_feats)
    assert torch.allclose(output_custom, output_dgl, atol=1e-6)

    # Compare gradients
    output_custom.sum().backward(retain_graph=True)
    output_dgl.sum().backward()
    assert compare_gradients(gat_layer, gat_conv)


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

    # Convert adjacency matrix to torch sparse tensor
    adj_matrix = adj_matrix.to_sparse_coo()

    # Custom GATLayer
    gat_layer = GATLayer((in_feats_src, in_feats_dst), out_feats, num_heads,
                         residual=residual, bias=bias)
    output_custom = gat_layer((h_src, h_dst), adj_matrix, edge_weight=edge_weights)

    # DGL GATConv
    gat_conv = GATConv((in_feats_src, in_feats_dst), out_feats, num_heads,
                       feat_drop=0.0, attn_drop=0.0, negative_slope=0.2,
                       residual=residual, activation=None, bias=bias)
    copy_params(gat_layer, gat_conv)
    output_dgl = gat_conv(g, (h_src, h_dst), edge_weight=edge_weights)

    assert output_custom.shape == (num_dst_nodes, num_heads, out_feats)
    assert torch.allclose(output_custom, output_dgl, atol=1e-6)

    # Compare gradients
    output_custom.sum().backward(retain_graph=True)
    output_dgl.sum().backward()
    assert compare_gradients(gat_layer, gat_conv)


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

    # Convert adjacency matrix to torch sparse tensor
    adj_matrix = adj_matrix.to_sparse_coo()

    # Custom GATLayer
    gat_layer = GATLayer(in_feats, out_feats, num_heads, residual=residual, bias=bias,
                         allow_zero_in_degree=True)
    output_custom = gat_layer(h, adj_matrix)

    # DGL GATConv
    gat_conv = GATConv(in_feats, out_feats, num_heads, feat_drop=0.0, attn_drop=0.0,
                       negative_slope=0.2, residual=residual, activation=None,
                       allow_zero_in_degree=True)
    copy_params(gat_layer, gat_conv)
    output_dgl = gat_conv(g, h)

    assert output_custom.shape == (num_nodes, num_heads, out_feats)
    assert torch.allclose(output_custom, output_dgl, atol=1e-6)

    # Compare gradients
    output_custom.sum().backward(retain_graph=True)
    output_dgl.sum().backward()
    assert compare_gradients(gat_layer, gat_conv)


def test_han():

    import json
    with open('model_configs/HAN_ACM.json') as f:
        config = json.load(f)

    dataset = ACM("./data/ACM",
                  {"labels": "./data/ACM/labels.pkl", "check_data_size": True})

    han = HAN(config, dataset)

    # Test forward
    output = han.forward()

    # Training
    han.fit()

    summary = han.get_summary()

    print(summary)
