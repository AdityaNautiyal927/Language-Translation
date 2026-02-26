
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):

    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int, attention_dim: int = 256):
       
        super(BahdanauAttention, self).__init__()

        # Project encoder hidden states
        self.W_encoder = nn.Linear(encoder_hidden_dim, attention_dim, bias=False)
        # Project decoder hidden state
        self.W_decoder = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)
        # Reduce to a scalar energy per token
        self.V = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,   # (batch, decoder_hidden_dim)
        encoder_outputs: torch.Tensor,  # (batch, src_len, encoder_hidden_dim)
        src_mask: torch.Tensor = None,  # (batch, src_len) — True where valid
    ):
        
        batch_size, src_len, _ = encoder_outputs.size()

        # (batch, src_len, attention_dim)
        encoder_energy = self.W_encoder(encoder_outputs)

        # (batch, 1, attention_dim)  →  broadcast over src_len
        decoder_energy = self.W_decoder(decoder_hidden).unsqueeze(1)

        # Combined energy: (batch, src_len, attention_dim)
        combined = torch.tanh(encoder_energy + decoder_energy)

        # Scalar energy per position: (batch, src_len, 1) → (batch, src_len)
        energy = self.V(combined).squeeze(2)

        # Mask PAD positions before softmax
        if src_mask is not None:
            energy = energy.masked_fill(~src_mask, float("-inf"))

        # Normalise to probability distribution
        attention_weights = F.softmax(energy, dim=1)  # (batch, src_len)

        # Weighted sum of encoder outputs: (batch, encoder_hidden_dim)
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, src_len)
            encoder_outputs,                 # (batch, src_len, encoder_hidden_dim)
        ).squeeze(1)                         # (batch, encoder_hidden_dim)

        return context_vector, attention_weights



if __name__ == "__main__":
    batch, src_len = 4, 12
    enc_hidden_dim, dec_hidden_dim, attn_dim = 512, 512, 256

    encoder_outputs = torch.randn(batch, src_len, enc_hidden_dim)
    decoder_hidden  = torch.randn(batch, dec_hidden_dim)

 
    mask = torch.ones(batch, src_len, dtype=torch.bool)
    mask[:, -2:] = False

    attn = BahdanauAttention(enc_hidden_dim, dec_hidden_dim, attn_dim)
    ctx, weights = attn(decoder_hidden, encoder_outputs, mask)

    print(f"context_vector shape   : {ctx.shape}")       # (4, 512)
    print(f"attention_weights shape: {weights.shape}")   # (4, 12)
    print(f"weights sum to ~1      : {weights.sum(dim=1)}")  # each row sums to ≈1
    print("BahdanauAttention — OK")
