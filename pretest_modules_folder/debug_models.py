import torch
from models.han import GATConv as GATLayer, HAN
from datasets import ACM
import dgl
from dgl.nn.pytorch.conv import GATConv


def copy_params(src_layer, dst_layer):
    # Copy weights
    # for src_param, dst_param in zip(src_layer.parameters(), dst_layer.parameters()):
    #     if dst_param.requires_grad:
    #         dst_param.data = src_param.data.detach().clone()
    named_params_src = dict(src_layer.named_parameters())
    named_params_dst = dict(dst_layer.named_parameters())
    for name, param in named_params_src.items():
        named_params_dst[name].data = param.data.detach().clone()


def compare_gradients(tensor1, tensor2):
    return torch.allclose(tensor1.grad, tensor2.grad, atol=1e-6)


def debug_gat_layer():
    in_feats = 5
    out_feats = 3
    num_heads = 2
    num_nodes = 4

    residual = False
    bias = True

    h = torch.rand((num_nodes, in_feats))
    g = torch.tensor([[0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0]], dtype=torch.float32)
    g = g.to_sparse_coo()

    gat_layer = GATLayer(in_feats, out_feats, num_heads, residual=residual, bias=bias,
                         allow_zero_in_degree=True)
    output = gat_layer(h, g)

    assert output.shape == (num_nodes, out_feats * num_heads)


def debug_gat2():
    residual = False
    bias = False
    in_feats_src = 5
    in_feats_dst = 6
    out_feats = 3
    num_heads = 2
    num_src_nodes = 4
    num_dst_nodes = 3

    h_src = torch.rand((num_src_nodes, in_feats_src))
    h_dst = torch.rand((num_dst_nodes, in_feats_dst))
    g = torch.tensor([[1, 1, 0],
                      [1, 0, 1],
                      [0, 1, 1],
                      [0, 0, 1]], dtype=torch.float32)
    g = g.to_sparse_coo()

    gat_layer = GATLayer((in_feats_src, in_feats_dst), out_feats, num_heads,
                         residual=residual, bias=bias)
    output = gat_layer((h_src, h_dst), g)
    assert output.shape == (num_src_nodes, out_feats * num_heads)


def debug_gat_layer_bipartite_with_edge_weights():
    residual = False
    bias = False
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

    assert output_custom.shape == (num_dst_nodes, num_heads, out_feats)
    assert torch.allclose(output_custom, output_dgl, atol=1e-6)

    # Compare gradients
    output_custom.sum().backward(retain_graph=True)
    output_dgl.sum().backward()
    assert compare_gradients(h_src, h_src)
    assert compare_gradients(h_dst, h_dst)
    assert compare_gradients(edge_weights, edge_weights)


def debug_han():
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
