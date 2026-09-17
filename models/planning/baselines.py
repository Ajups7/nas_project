"""
Person C (Planning tier) - Step 6 of the build sequence: baseline
architectures. "SARIMA, seq2seq, TFT - written, not run" (per the build
sequence doc). PLAN.md's Phase 2 goal is a working, evaluable baseline
before any novel component (FAOC-Loss, MoE routing, the Regime Transformer)
gets layered in - these three exist and are mechanically validated with
synthetic inputs here (see validate_baselines.py); none are trained or run
against real data yet. Real training needs the team-agreed temporal split
(TEAM_PLAN.md's proposed section) to actually exist first.

Two different kinds of "baseline" here, deliberately not forced into one
shared interface:

- SARIMABaseline wraps statsmodels' SARIMAX - a classical statistical
  model that fits a single univariate time series (e.g. a daily disruption
  rate at one airport) directly. It has a fit()/forecast() interface, not
  a tensor-in/tensor-out one, because that's what SARIMA actually is -
  forcing it into a neural-style forward(x) would misrepresent it.
- Seq2SeqBaseline and TFTBaseline are PyTorch modules sharing one
  interface: forward(x) where x is (batch, seq_len, num_features) - a
  window of `seq_len` days' feature vectors (features/planning.py's
  output) - producing (batch,), a disruption probability. Same output
  shape as models/planning/moe_router.py's MoERegimeRouter, so it plugs
  into FAOCLoss the same way.

TFTBaseline is a SIMPLIFIED Temporal Fusion Transformer: a learned
variable-selection gate over input features, an LSTM encoder, one
multi-head self-attention block, then a linear head on the final
timestep. It captures TFT's core ideas (learned per-feature weighting +
attention over time) without the full published architecture's
multi-horizon quantile outputs or static-covariate encoders - this is a
benchmark to beat, not the project's actual novel contribution (that's the
Regime Transformer, Step 8).
"""

import pandas as pd
import torch
import torch.nn as nn
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMABaseline:
    """Classical seasonal ARIMA baseline over a single univariate series.
    order/seasonal_order default to a weekly-seasonality guess (period=7,
    matching a daily-granularity series) - Phase 2's actual baseline run
    should tune these against real data, not trust these defaults blindly.
    """

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self._fitted = None

    def fit(self, series: pd.Series) -> "SARIMABaseline":
        model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._fitted = model.fit(disp=False)
        return self

    def forecast(self, steps: int) -> pd.Series:
        if self._fitted is None:
            raise RuntimeError("call fit() before forecast()")
        return self._fitted.forecast(steps=steps)


class Seq2SeqBaseline(nn.Module):
    """LSTM encoder-decoder. Encodes a (batch, seq_len, num_features)
    window, then unrolls the decoder for `decode_steps` steps (default 1,
    since the Planning label - features/planning_label.py - is a single
    probability, not a sequence). Kept as a genuine encoder/decoder pair
    rather than collapsing to one LSTM, since "seq2seq" specifically names
    that architecture as one of the paper's Section 5 baselines.
    """

    def __init__(self, num_features: int, hidden_dim: int = 32, decode_steps: int = 1):
        super().__init__()
        self.decode_steps = decode_steps
        self.encoder = nn.LSTM(num_features, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(1, hidden_dim, batch_first=True)
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        _, (h, c) = self.encoder(x)
        decoder_input = torch.zeros(batch_size, self.decode_steps, 1, device=x.device)
        decoder_out, _ = self.decoder(decoder_input, (h, c))
        last_step = decoder_out[:, -1, :]
        logit = self.output_head(last_step).squeeze(-1)
        return torch.sigmoid(logit)


class TFTBaseline(nn.Module):
    """Simplified Temporal Fusion Transformer - see module docstring for
    exactly what's kept (variable selection, LSTM encoding, self-attention)
    and what's deliberately left out relative to the full published
    architecture."""

    def __init__(self, num_features: int, hidden_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.variable_selection = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Softmax(dim=-1),
        )
        self.encoder = nn.LSTM(num_features, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, return_variable_weights: bool = False):
        weights = self.variable_selection(x)  # (batch, seq_len, num_features), rows sum to 1
        selected = x * weights
        encoded, _ = self.encoder(selected)  # (batch, seq_len, hidden_dim)
        attended, _ = self.attention(encoded, encoded, encoded)
        last_step = attended[:, -1, :]
        logit = self.output_head(last_step).squeeze(-1)
        pred_probs = torch.sigmoid(logit)

        if return_variable_weights:
            return pred_probs, weights
        return pred_probs
