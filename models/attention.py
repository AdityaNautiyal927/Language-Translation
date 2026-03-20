
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention.

    Computes a context vector as a weighted sum of encoder outputs,
    where weights are determined by the compatibility between each
    encoder position and the current decoder hidden state.
    """

    def __init__(
        self,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        attention_dim: int = 256,
    ):
        super(BahdanauAttention, self).__init__()

        # Project encoder hidden states → attention_dim
        self.W_encoder = nn.Linear(encoder_hidden_dim, attention_dim, bias=False)
        # Project decoder hidden state → attention_dim
        self.W_decoder = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)
        # Reduce combined energy to a scalar per token
        self.V = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,          # (batch, decoder_hidden_dim)
        encoder_outputs: torch.Tensor,          # (batch, src_len, encoder_hidden_dim)
        src_mask: Optional[torch.Tensor] = None,  # (batch, src_len) — True where valid
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # (batch, src_len, attention_dim)
        encoder_energy = self.W_encoder(encoder_outputs)

        # (batch, 1, attention_dim) — broadcasts over src_len
        decoder_energy = self.W_decoder(decoder_hidden).unsqueeze(1)

        # Combined energy: (batch, src_len, attention_dim)
        combined = torch.tanh(encoder_energy + decoder_energy)

        # Scalar energy per position: (batch, src_len)
        energy = self.V(combined).squeeze(2)

        # Mask PAD positions before softmax — set them to -inf
        if src_mask is not None:
            energy = energy.masked_fill(~src_mask, float("-inf"))

        # Normalise to probability distribution
        attention_weights = F.softmax(energy, dim=1)  # (batch, src_len)

        # Fix #11 & #12: replace NaN with 0 to prevent silent propagation
        # when a row had all -inf energies (e.g. fully-padded src sequence).
        attention_weights = attention_weights.nan_to_num(0.0)

        # Weighted sum of encoder outputs: (batch, encoder_hidden_dim)
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, src_len)
            encoder_outputs,                 # (batch, src_len, encoder_hidden_dim)
        ).squeeze(1)                         # (batch, encoder_hidden_dim)

        return context_vector, attention_weights


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    batch, src_len = 4, 12
    enc_hidden_dim, dec_hidden_dim, attn_dim = 512, 512, 256

    encoder_outputs = torch.randn(batch, src_len, enc_hidden_dim)
    decoder_hidden  = torch.randn(batch, dec_hidden_dim)

    # Mask: last 2 positions are PAD
    mask = torch.ones(batch, src_len, dtype=torch.bool)
    mask[:, -2:] = False

    attn = BahdanauAttention(enc_hidden_dim, dec_hidden_dim, attn_dim)
    ctx, weights = attn(decoder_hidden, encoder_outputs, mask)

    print(f"context_vector shape   : {ctx.shape}")       # (4, 512)
    print(f"attention_weights shape: {weights.shape}")   # (4, 12)
    print(f"weights sum to ~1      : {weights.sum(dim=1)}")  # each ≈ 1.0

    # Extra: test all-PAD edge case (should not produce NaN)
    all_pad_mask = torch.zeros(batch, src_len, dtype=torch.bool)
    ctx2, w2 = attn(decoder_hidden, encoder_outputs, all_pad_mask)
    assert not torch.isnan(ctx2).any(), "NaN detected in context vector!"
    print("All-PAD edge case      : OK (no NaN)")

    print("BahdanauAttention — OK")
