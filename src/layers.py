import torch
import torch.nn as nn


def init_uniform_param(shape, hidden_size):
    """Create a trainable parameter initialised uniformly in [-1/sqrt(H), 1/sqrt(H)]."""
    bound = 1.0 / (hidden_size ** 0.5)
    tensor = torch.empty(shape, dtype=torch.float32)
    tensor.uniform_(-bound, bound)
    return nn.Parameter(tensor)


class RNNCell(nn.Module):
    """Plain recurrent cell: h_t = tanh(W x_t + U h_{t-1} + b)."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = init_uniform_param((hidden_size, input_size), hidden_size)
        self.U = init_uniform_param((hidden_size, hidden_size), hidden_size)
        self.b = init_uniform_param((hidden_size,), hidden_size)

    def forward(self, x_t, h_prev):
        # x_t: (B, D), h_prev: (B, H)
        linear_sum = torch.matmul(x_t, self.W.t()) + torch.matmul(h_prev, self.U.t()) + self.b
        h_t = torch.tanh(linear_sum)
        return h_t


class LSTMCell(nn.Module):
    """LSTM cell from gate equations with stacked [input, forget, candidate, output] weights."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        H = hidden_size

        self.W = init_uniform_param((4 * H, input_size), H)
        self.U = init_uniform_param((4 * H, H), H)
        self.b = init_uniform_param((4 * H,), H)

    def forward(self, x_t, h_prev, c_prev):
        # Compute W x_t + U h_{t-1} + b of shape (B, 4H)
        H = self.hidden_size
        combined = torch.matmul(x_t, self.W.t()) + torch.matmul(h_prev, self.U.t()) + self.b

        # Slice explicitly in stacked gate order: input, forget, cell candidate, output
        raw_i = combined[:, 0:H]
        raw_f = combined[:, H : 2 * H]
        raw_g = combined[:, 2 * H : 3 * H]
        raw_o = combined[:, 3 * H : 4 * H]

        i_t = torch.sigmoid(raw_i)
        f_t = torch.sigmoid(raw_f)
        g_t = torch.tanh(raw_g)
        o_t = torch.sigmoid(raw_o)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class LSTM(nn.Module):
    """Unroll LSTMCell across all T time steps from h_0 = 0, c_0 = 0."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x):
        # x has shape (B, T, D)
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        h_t = torch.zeros((batch_size, self.hidden_size), dtype=x.dtype, device=x.device)
        c_t = torch.zeros((batch_size, self.hidden_size), dtype=x.dtype, device=x.device)

        hidden_states = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t, c_t = self.cell(x_t, h_t, c_t)
            hidden_states.append(h_t)

        # Stack along time axis -> (B, T, H)
        return torch.stack(hidden_states, dim=1)


class BiLSTM(nn.Module):
    """Bidirectional LSTM built from two independent LSTM layers (forward and backward)."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fwd = LSTM(input_size, hidden_size)
        self.bwd = LSTM(input_size, hidden_size)

    def forward(self, x):
        # Forward pass reads x_1 .. x_T
        h_forward = self.fwd(x)

        # Manually reverse sequence along time axis: x_T .. x_1
        seq_len = x.shape[1]
        rev_idx = [seq_len - 1 - t for t in range(seq_len)]
        x_reversed = x[:, rev_idx, :]

        # Backward pass on reversed input, then reverse outputs back so step t aligns with x_t
        h_backward_rev = self.bwd(x_reversed)
        h_backward = h_backward_rev[:, rev_idx, :]

        # Concatenate [h_fwd_t ; h_bwd_t] at every time step -> (B, T, 2H)
        return torch.cat([h_forward, h_backward], dim=2)


class Attention(nn.Module):
    """Additive temporal attention pooling: e_t = v^T tanh(Wa h_t + ba), alpha = softmax(e)."""

    def __init__(self, in_size, att_size=64):
        super().__init__()
        self.in_size = in_size
        self.att_size = att_size

        self.Wa = init_uniform_param((att_size, in_size), att_size)
        self.ba = init_uniform_param((att_size,), att_size)
        self.v = init_uniform_param((att_size,), att_size)

    def forward(self, h):
        # h: (B, T, 2H)
        # Wa h_t + ba -> (B, T, 64)
        u_t = torch.tanh(torch.matmul(h, self.Wa.t()) + self.ba)

        # Dot product with v -> scalar score e_t per time step: (B, T)
        e_t = torch.matmul(u_t, self.v)

        # Normalise over time steps t = 1..128
        alpha = torch.softmax(e_t, dim=1)

        # Weighted sum c = sum_{t=1}^T alpha_t * h_t -> (B, 2H)
        context = torch.sum(alpha.unsqueeze(2) * h, dim=1)
        return context, alpha


if __name__ == "__main__":
    torch.manual_seed(0)
    dummy_x = torch.randn(8, 128, 9)

    rnn_cell = RNNCell(9, 64)
    h0 = torch.zeros(8, 64)
    h1 = rnn_cell(dummy_x[:, 0, :], h0)
    print("RNNCell single-step output shape:", tuple(h1.shape))

    lstm_layer = LSTM(9, 64)
    out_lstm = lstm_layer(dummy_x)
    print("LSTM sequence output shape:      ", tuple(out_lstm.shape))

    bilstm_layer = BiLSTM(9, 64)
    out_bilstm = bilstm_layer(dummy_x)
    print("BiLSTM sequence output shape:    ", tuple(out_bilstm.shape))

    att_layer = Attention(128, att_size=64)
    ctx, weights = att_layer(out_bilstm)
    print("Attention context & alpha shapes:", tuple(ctx.shape), tuple(weights.shape))
    print("Attention weights sum per window:", weights.sum(dim=1).detach().tolist()[:3])
