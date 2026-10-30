"""Comfi-FastGRNN: FastGRNN with trainable complementary filter (gamma, lambda).

基于 Fast-ULCNet 固定版本复制并适配；commit、适配范围和 MIT 许可见本目录
``PROVENANCE`` 与 ``LICENSE.Fast-ULCNet``。
Paper: Fast-ULCNet, Eq. (4) h_t_comfi = gamma*h_t + (1-gamma)*lambda
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gen_non_linearity(A: torch.Tensor, non_linearity: str):
    if non_linearity == "tanh":
        return torch.tanh(A)
    if non_linearity == "sigmoid":
        return torch.sigmoid(A)
    if non_linearity == "relu":
        return torch.relu(A)
    if non_linearity == "quantTanh":
        return torch.clamp(A, -1.0, 1.0)
    if non_linearity == "quantSigm":
        return torch.clamp((A + 1.0) / 2.0, 0.0, 1.0)
    if non_linearity == "quantSigm4":
        return torch.clamp((A + 2.0) / 4.0, 0.0, 1.0)
    if callable(non_linearity):
        return non_linearity(A)
    raise ValueError(
        "non_linearity must be one of ['tanh', 'sigmoid', 'relu', "
        "'quantTanh', 'quantSigm', 'quantSigm4'] or callable"
    )


class ComfiFastGRNNCell(nn.Module):
    """Comfi-FastGRNN cell: FastGRNN update + complementary filter."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        gate_non_linearity: str = "sigmoid",
        update_non_linearity: str = "tanh",
        w_rank: int | None = None,
        u_rank: int | None = None,
        zeta_init: float = 1.0,
        nu_init: float = -4.0,
        lambda_init: float = 0.0,
        gamma_init: float = 0.999,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        if w_rank is None:
            self.w_matrix = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        else:
            self.w_matrix_1 = nn.Parameter(torch.randn(input_size, w_rank) * 0.1)
            self.w_matrix_2 = nn.Parameter(torch.randn(w_rank, hidden_size) * 0.1)
        if u_rank is None:
            self.u_matrix = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        else:
            self.u_matrix_1 = nn.Parameter(torch.randn(hidden_size, u_rank) * 0.1)
            self.u_matrix_2 = nn.Parameter(torch.randn(u_rank, hidden_size) * 0.1)
        self.bias_gate = nn.Parameter(torch.ones(1, hidden_size))
        self.bias_update = nn.Parameter(torch.ones(1, hidden_size))
        self.zeta = nn.Parameter(torch.tensor([[zeta_init]], dtype=torch.float32))
        self.nu = nn.Parameter(torch.tensor([[nu_init]], dtype=torch.float32))
        self.lambd = nn.Parameter(torch.tensor([lambda_init], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor([gamma_init], dtype=torch.float32))
        self._gate_nl = gate_non_linearity
        self._update_nl = update_non_linearity
        self._w_rank = w_rank
        self._u_rank = u_rank

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        W = x @ self.w_matrix if self._w_rank is None else x @ self.w_matrix_1 @ self.w_matrix_2
        U = h_prev @ self.u_matrix if self._u_rank is None else h_prev @ self.u_matrix_1 @ self.u_matrix_2
        z = _gen_non_linearity(W + U + self.bias_gate, self._gate_nl)
        h_hat = _gen_non_linearity(W + U + self.bias_update, self._update_nl)
        h = z * h_prev + (torch.sigmoid(self.zeta) * (1 - z) + torch.sigmoid(self.nu)) * h_hat
        gamma_clamped = torch.clamp(self.gamma, 0.0, 1.0)
        return gamma_clamped * h + (1 - gamma_clamped) * self.lambd


class ComfiFastGRNN(nn.Module):
    """Comfi-FastGRNN layer (single or multi-layer, optional bidirectional)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        gate_non_linearity: str = "sigmoid",
        update_non_linearity: str = "tanh",
        w_rank: int | None = None,
        u_rank: int | None = None,
        zeta_init: float = 1.0,
        nu_init: float = -4.0,
        lambda_init: float = 0.0,
        gamma_init: float = 0.999,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.cells_fwd = nn.ModuleList()
        self.cells_bwd = nn.ModuleList() if bidirectional else None

        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size * self.num_directions
            self.cells_fwd.append(
                ComfiFastGRNNCell(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    gate_non_linearity=gate_non_linearity,
                    update_non_linearity=update_non_linearity,
                    w_rank=w_rank,
                    u_rank=u_rank,
                    zeta_init=zeta_init,
                    nu_init=nu_init,
                    lambda_init=lambda_init,
                    gamma_init=gamma_init,
                )
            )
            if bidirectional:
                self.cells_bwd.append(
                    ComfiFastGRNNCell(
                        input_size=in_size,
                        hidden_size=hidden_size,
                        gate_non_linearity=gate_non_linearity,
                        update_non_linearity=update_non_linearity,
                        w_rank=w_rank,
                        u_rank=u_rank,
                        zeta_init=zeta_init,
                        nu_init=nu_init,
                        lambda_init=lambda_init,
                        gamma_init=gamma_init,
                    )
                )

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        if not self.batch_first:
            x = x.transpose(0, 1)
        batch_size, seq_len, _ = x.size()
        if h0 is None:
            h0 = x.new_zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        layer_input = x
        h_n = []
        for layer in range(self.num_layers):
            fw_cell = self.cells_fwd[layer]
            h_fw = h0[layer * self.num_directions + 0]
            fw_outs = []
            for t in range(seq_len):
                h_fw = fw_cell(layer_input[:, t, :], h_fw)
                fw_outs.append(h_fw.unsqueeze(1))
            fw_out = torch.cat(fw_outs, dim=1)
            if self.bidirectional:
                bw_cell = self.cells_bwd[layer]
                h_bw = h0[layer * self.num_directions + 1]
                bw_outs = []
                for t in reversed(range(seq_len)):
                    h_bw = bw_cell(layer_input[:, t, :], h_bw)
                    bw_outs.append(h_bw.unsqueeze(1))
                bw_outs.reverse()
                bw_out = torch.cat(bw_outs, dim=1)
                layer_out = torch.cat([fw_out, bw_out], dim=2)
                h_n.extend([h_fw, h_bw])
            else:
                layer_out = fw_out
                h_n.append(h_fw)
            if self.dropout > 0.0 and layer < self.num_layers - 1:
                layer_out = F.dropout(layer_out, p=self.dropout, training=self.training)
            layer_input = layer_out
        h_n = torch.stack(h_n, dim=0)
        if not self.batch_first:
            layer_input = layer_input.transpose(0, 1)
        return layer_input, h_n
