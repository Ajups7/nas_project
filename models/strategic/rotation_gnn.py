"""
Person B (Strategic tier) - Step 5 of the build sequence: the
spatio-temporal GNN (see TEAM_PLAN.md: "Strategic baseline model, then the
spatio-temporal GNN work in Phase 3 (rotation-chain edges fit naturally at
this horizon, where next-day network effects matter most)"). This is the
Strategic tier's actual Phase 3 novel architecture (PLAN.md item 6),
analogous to Person C's SectorGraphAttentionRegimeTransformer for the
Planning tier - a graph-attention architecture representing disruption
propagation over scheduled aircraft rotations rather than geographic
proximity alone.

Unlike models/planning/regime_transformer.py, this one is NOT blocked on a
missing graph: features/rotation_graph.py's build_rotation_adjacency()
(Step 1) already exists and is validated against real BTS data. That
module's own validation doc says explicitly "this can only be tested [with
a real graph] once features/rotation_graph.py exists" - it now does, and
validate_rotation_gnn.py below is exactly that real-graph test, in
addition to the synthetic mechanism check every prior architecture in this
project starts with.

Design - genuinely "spatio-temporal," not "temporal-then-spatial":
regime_transformer.py's design fully encodes each node's time series
first (an LSTM run to completion), then does ONE spatial attention pass
afterward - space and time are two separate, sequential stages. This
module interleaves them instead: at every one of the seq_len daily
timesteps, each node's hidden state is (1) updated from that day's own
features via a GRU cell, THEN (2) mixed with its rotation-chain neighbors'
hidden states via masked graph attention, and that mixed state becomes the
input to the NEXT timestep's GRU update. A disruption signal can
therefore propagate outward across the rotation-chain graph and forward in
time in the same pass - e.g. airport A's day-3 conditions can influence
airport B's day-4 hidden state if A and B share a rotation-chain edge,
which a fully-separate "encode all days, then look at space once"
pipeline can't represent. This is the actual "network effects" TEAM_PLAN.md
calls out as the reason rotation-chain edges "fit naturally at this
horizon."

Graph attention, not a literal GCN: same choice as regime_transformer.py,
and for the same reason - build_attention_mask() (reused directly from
that module rather than reimplemented, since it's a pure, tier-agnostic
function: adjacency in, additive attention mask out) turns "attend only to
rotation-chain neighbors" into an ordinary masked-softmax attention
problem, without needing separate degree-normalization the way a raw
adjacency-weighted graph convolution would.

No training happens here - same caveat as every prior architecture in this
project: mechanically validated (synthetic AND, for the first time in this
project, real graph + real features), not trained, gated on the
team-agreed temporal split (TEAM_PLAN.md).
"""

import torch
import torch.nn as nn

from models.planning.regime_transformer import build_attention_mask


class RotationChainSpatioTemporalGNN(nn.Module):
    """
    x: (batch, num_nodes, seq_len, num_features) - a window of daily
       Strategic-tier feature vectors (features/strategic.py's
       build_strategic_features() output) for each of num_nodes airports.
    adjacency: (num_nodes, num_nodes) - rotation-chain edges, from
       features/rotation_graph.py's build_rotation_adjacency() (Step 1).

    Returns pred_probs, shape (batch, num_nodes) - one disruption
    probability per airport in the graph.
    """

    def __init__(self, num_features: int, hidden_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(num_features, hidden_dim)
        self.graph_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, return_attention_weights: bool = False):
        batch, num_nodes, seq_len, num_features = x.shape
        attn_mask = build_attention_mask(adjacency)

        h = torch.zeros(batch * num_nodes, self.hidden_dim, device=x.device, dtype=x.dtype)
        attn_weights = None
        for t in range(seq_len):
            x_t = x[:, :, t, :].reshape(batch * num_nodes, num_features)
            h = self.cell(x_t, h)  # (batch*num_nodes, hidden_dim) - temporal update

            h_nodes = h.reshape(batch, num_nodes, self.hidden_dim)
            attended, attn_weights = self.graph_attention(
                h_nodes, h_nodes, h_nodes, attn_mask=attn_mask
            )  # spatial update - mixes each node with its rotation-chain neighbors
            h = attended.reshape(batch * num_nodes, self.hidden_dim)

        h_nodes = h.reshape(batch, num_nodes, self.hidden_dim)
        logits = self.output_head(h_nodes).squeeze(-1)  # (batch, num_nodes)
        pred_probs = torch.sigmoid(logits)

        if return_attention_weights:
            return pred_probs, attn_weights
        return pred_probs
