import math
import functools

import torch

import emphases

class Encoder(torch.nn.Module):

    def __init__(
        self,
        input_channels=emphases.NUM_FEATURES,
        output_channels=1,
        hidden_channels=emphases.HIDDEN_CHANNELS,
        n_heads=emphases.N_HEADS,
        n_layers=emphases.N_LAYERS,
        window_size=4):
        super().__init__()

        filter_channels = hidden_channels
        kernel_size = emphases.FFN_KERNEL_SIZE

        conv_fn = functools.partial(
            torch.nn.Conv1d,
            kernel_size=emphases.ATTN_ENC_KERNEL_SIZE,
            padding='same'
        )
        self.pre_encoder = torch.nn.Sequential(
            conv_fn(input_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, hidden_channels)
        )
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        self.decoder = torch.nn.Sequential(
            conv_fn(hidden_channels, output_channels)
        )
        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    window_size=window_size))
            self.norm_layers_1.append(
                LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size))
            self.norm_layers_2.append(
                LayerNorm(hidden_channels))

    def forward(self, x, unused_bounds, unused_lengths, mask=None):
        if mask is not None:
            mask_unsqueezed = mask.unsqueeze(2) * mask.unsqueeze(-1) 
            x = x * mask
        else:
            mask_unsqueezed = None
        x = self.pre_encoder(x)
        for i in range(len(self.attn_layers)):
            y = self.attn_layers[i](
                x,
                x,
                mask_unsqueezed)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, mask)
            x = self.norm_layers_2[i](x + y)
        return self.decoder(x)


class MultiHeadAttention(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        channels,
        out_channels,
        n_heads,
        window_size=4):
        super().__init__()
        assert channels % n_heads == 0
        # Setup layers
        self.n_heads = n_heads
        self.window_size = window_size
        self.conv_q = torch.nn.Conv1d(in_channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(in_channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(in_channels, channels, 1)
        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)

        # Setup relative positional embedding
        self.k_channels = channels // n_heads
        rel_stddev = self.k_channels ** -0.5
        self.emb_rel_k = torch.nn.Parameter(
            rel_stddev * torch.randn(
                1,
                window_size * 2 + 1,
                self.k_channels))
        self.emb_rel_v = torch.nn.Parameter(
            rel_stddev * torch.randn(
                1,
                window_size * 2 + 1,
                self.k_channels))

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=mask)
        return self.conv_o(x)

    def attention(self, query, key, value, mask=None):
        # Reshape (batch, channels, time) ->
        #         (batch, heads, time, channels // heads)
        batch, channels, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(
            batch,
            self.n_heads,
            self.k_channels,
            t_t).transpose(2, 3)
        key = key.view(
            batch,
            self.n_heads,
            self.k_channels,
            t_s).transpose(2, 3)
        value = value.view(
            batch,
            self.n_heads,
            self.k_channels,
            t_s).transpose(2, 3)

        # Compute attention matrix
        scores = torch.matmul(
            query / math.sqrt(self.k_channels),
            key.transpose(-2, -1))

        # Relative positional representation
        relative_embeddings = self.relative_embeddings(self.emb_rel_k, t_s)
        relative_logits = torch.matmul(
            query / math.sqrt(self.k_channels),
            relative_embeddings.unsqueeze(0).transpose(-2, -1))
        scores += self.relative_to_absolute(relative_logits)

        # Apply sequence mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        # Compute output activation
        # (batch, heads, t_t, t_s)
        attention = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attention, value)

        # Convert to absolute positional representation to adjust output
        output += torch.matmul(
            self.absolute_to_relative(attention),
            self.relative_embeddings(self.emb_rel_v, t_s).unsqueeze(0))

        # Reshape (batch, heads, time, channels // heads) ->
        #         (batch, channels, time]
        output = output.transpose(2, 3).contiguous().view(batch, channels, t_t)

        return output, attention

    def relative_embeddings(self, embedding, length):
        # Pad first before slice to avoid using cond ops
        pad_length = max(length - (self.window_size + 1), 0)
        start = max((self.window_size + 1) - length, 0)
        end = start + 2 * length - 1
        if pad_length > 0:
            padded_embedding = torch.nn.functional.pad(
                embedding,
                (0, 0, pad_length, pad_length))
        else:
            padded_embedding = embedding
        return padded_embedding[:, start:end]

    def relative_to_absolute(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()

        # Concat columns of pad to shift from relative to absolute indexing
        x = torch.nn.functional.pad(x, (0, 1))

        # Concat extra elements so to add up to shape (len + 1, 2 * len - 1)
        x_flat = x.view([batch, heads, 2 * length * length])
        x_flat = torch.nn.functional.pad(x_flat, (0, length - 1))

        # Reshape and slice out the padded elements.
        shape = (batch, heads, length + 1, 2 * length - 1)
        return x_flat.view(shape)[:, :, :length, length - 1:]

    def absolute_to_relative(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()

        # Pad along column
        x = torch.nn.functional.pad(x, (0, length - 1))
        x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])

        # Add 0's in the beginning that will skew the elements after reshape
        x_flat = torch.nn.functional.pad(x_flat, (length, 0))
        return x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]

class LayerNorm(torch.nn.Module):

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return torch.nn.functional.layer_norm(
            x.transpose(1, -1),
            (self.channels,),
            self.gamma,
            self.beta,
            self.eps).transpose(1, -1)

class FFN(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_1 = torch.nn.Conv1d(
            in_channels,
            filter_channels,
            kernel_size)
        self.conv_2 = torch.nn.Conv1d(
            filter_channels,
            out_channels,
            kernel_size)

    def forward(self, x, mask=None):
        if mask is not None: x = x * mask
        x = self.conv_1(self.pad(x))
        x = torch.relu(x)
        if mask is not None: x = x * mask
        x = self.conv_2(self.pad(x))
        if mask is not None: x = x * mask
        return x

    def pad(self, x):
        if self.kernel_size == 1:
            return x
        padding = ((self.kernel_size - 1) // 2, self.kernel_size // 2)
        return torch.nn.functional.pad(x, padding)