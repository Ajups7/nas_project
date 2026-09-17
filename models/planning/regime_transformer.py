"""
Person C (Planning tier) - Step 8 of the build sequence: the Regime
Transformer stub. "Sector-graph-attention architecture — written,
untrained." This is the project's actual Phase 3 novel architecture for
this tier (PLAN.md item 6), distinct from Step 2's MoE regime routing
(PLAN.md item 7) - the two are separate novel components, not one built on
top of the other.

CROSS-TEAM DEPENDENCY, not yet available: PLAN.md frames this
architecture as representing "disruption propagation over scheduled
aircraft rotations rather than geographic proximity alone" - i.e. it needs
a graph of rotation-chain edges between airports (built from BTS
tail-number sequencing). That's explicitly Person B's shared utility
(TEAM_PLAN.md: `features/rotation_graph.py`), and it does not exist in this
repo yet (checked directly - no file matching "rotation" anywhere).

Rather than block on that, this module takes an `adjacency` matrix as a
plain input argument - any (num_airports, num_airports) tensor, 1 where a
rotation-chain edge exists. Person B's real module, once it exists, only
needs to produce that shape; nothing else here needs to change. Validated
here (see regime_transformer_validation.md) with a synthetic, made-up
adjacency matrix - NOT Person B's real rotation-chain structure - purely to
prove the attention mechanism actually respects whatever graph it's given.

Naming note: PLAN.md calls this "sector-graph-attention," but the actual
edges are airport-to-airport rotation-chain edges (Person B's module), not
literal ATC airspace sectors. Kept as the paper's own name for the
architecture rather than silently renamed.

Two attention mechanisms, doing two different jobs:
1. Graph attention - each airport's encoded time-series representation
   attends only to airports it has a rotation-chain edge to (plus itself,
   a standard GNN self-loop convention), rather than to every other
   airport indiscriminately. This is the "propagation over rotations, not
   geography" part.
2. Regime attention - each airport's post-graph representation attends
   over 4 learned regime embeddings (normal/gdp/ground_stop/mit - the same
   4 regimes as models/planning/moe_router.py, reused for consistency, not
   duplicated as new terminology).

No training happens here - same caveat as every prior architecture in this
project: mechanically validated with synthetic inputs, not run against
real data, gated on both the team-agreed temporal split (TEAM_PLAN.md) and
Person B's real rotation-chain graph.
"""

import torch
import torch.nn as nn

from models.planning.moe_router import REGIME_NAMES


def build_attention_mask(adjacency: torch.Tensor) -> torch.Tensor:
    """adjacency: (num_nodes, num_nodes), nonzero where a rotation-chain
    edge exists. Self-loops are added automatically - a standard GNN
    convention, without which an airport with zero rotation-chain
    connections would attend to nothing and get an undefined (all -inf)
    softmax row. Returns an additive float mask: 0.0 where attention is
    allowed, -inf where masked out."""
    num_nodes = adjacency.shape[0]
    allowed = (adjacency > 0) | torch.eye(num_nodes, dtype=torch.bool, device=adjacency.device)
    mask = torch.zeros(num_nodes, num_nodes, device=adjacency.device)
    return mask.masked_fill(~allowed, float("-inf"))


class SectorGraphAttentionRegimeTransformer(nn.Module):
    """
    x: (batch, num_nodes, seq_len, num_features) - a window of daily
       feature vectors (features/planning.py's output) for each of
       num_nodes airports.
    adjacency: (num_nodes, num_nodes) - rotation-chain edges. Placeholder
       synthetic data until Person B's features/rotation_graph.py exists.

    Returns pred_probs, shape (batch, num_nodes) - one disruption
    probability per airport in the graph.
    """

    def __init__(self, num_features: int, hidden_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temporal_encoder = nn.LSTM(num_features, hidden_dim, batch_first=True)
        self.graph_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.regime_embeddings = nn.Parameter(torch.randn(len(REGIME_NAMES), hidden_dim))
        self.regime_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, return_attention_weights: bool = False):
        batch, num_nodes, seq_len, num_features = x.shape

        # Encode each airport's time series independently.
        x_flat = x.reshape(batch * num_nodes, seq_len, num_features)
        _, (h, _) = self.temporal_encoder(x_flat)
        node_embed = h[-1].reshape(batch, num_nodes, self.hidden_dim)

        # Graph attention: airports attend only to rotation-chain neighbors + self.
        attn_mask = build_attention_mask(adjacency)
        graph_out, graph_attn_weights = self.graph_attention(
            node_embed, node_embed, node_embed, attn_mask=attn_mask
        )

        # Regime attention: each airport attends over the 4 learned regime embeddings.
        regime_kv = self.regime_embeddings.unsqueeze(0).expand(batch, -1, -1)
        regime_out, regime_attn_weights = self.regime_attention(graph_out, regime_kv, regime_kv)

        logits = self.output_head(regime_out).squeeze(-1)  # (batch, num_nodes)
        pred_probs = torch.sigmoid(logits)

        if return_attention_weights:
            return pred_probs, graph_attn_weights, regime_attn_weights
        return pred_probs
