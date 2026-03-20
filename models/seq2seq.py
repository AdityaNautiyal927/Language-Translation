
from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn

from models.attention import BahdanauAttention


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class Encoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
        pad_idx: int = 0,
    ):
        super(Encoder, self).__init__()
        self.hidden_dim     = hidden_dim
        self.num_layers     = num_layers
        self.bidirectional  = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)

        if bidirectional:
            self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src: torch.Tensor):
        # src: (batch, src_len)
        embedded = self.dropout(self.embedding(src))

        # encoder_outputs: (batch, src_len, hidden_dim * num_directions)
        # hidden:          (num_layers * num_directions, batch, hidden_dim)
        encoder_outputs, hidden = self.gru(embedded)

        if self.bidirectional:
            # Fix #7: correctly handle multi-layer bidirectional hidden state.
            # hidden shape: (num_layers * 2, batch, hidden_dim)
            # Reshape to   (num_layers, 2, batch, hidden_dim)
            batch_size = hidden.size(1)
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
            # Take only the LAST layer's two directions
            fwd = hidden[-1, 0, :, :]   # (batch, hidden_dim)
            bwd = hidden[-1, 1, :, :]   # (batch, hidden_dim)
            hidden = torch.tanh(self.fc_hidden(torch.cat([fwd, bwd], dim=1)))
            # hidden: (batch, hidden_dim)
        else:
            # Take the last layer's hidden state
            hidden = hidden[-1, :, :]   # (batch, hidden_dim)

        return encoder_outputs, hidden


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class Decoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        encoder_hidden_dim: int,
        hidden_dim: int,
        attention_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers   # store for use in forward

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(encoder_hidden_dim, hidden_dim, attention_dim)
        self.gru = nn.GRU(
            embedding_dim + encoder_hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim + encoder_hidden_dim + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt_token: torch.Tensor,        # (batch,)
        hidden: torch.Tensor,           # (batch, hidden_dim)  — last layer only
        encoder_outputs: torch.Tensor,  # (batch, src_len, encoder_hidden_dim)
        src_mask: torch.Tensor = None,  # (batch, src_len)
    ):
        embedded = self.dropout(self.embedding(tgt_token.unsqueeze(1)))
        # embedded: (batch, 1, embedding_dim)

        context, attn_weights = self.attention(hidden, encoder_outputs, src_mask)
        # context: (batch, encoder_hidden_dim)

        gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        # gru_input: (batch, 1, embedding_dim + encoder_hidden_dim)

        # Fix #8: expand hidden to (num_layers, batch, hidden_dim) for GRU.
        # We repeat the same vector across all layers so that the decoder
        # is seeded from the encoder summary at every layer.
        hidden_3d = hidden.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        gru_out, new_hidden_3d = self.gru(gru_input, hidden_3d)
        # Use the last layer's output as the new hidden summary
        new_hidden = new_hidden_3d[-1]  # (batch, hidden_dim)

        gru_out_sq  = gru_out.squeeze(1)    # (batch, hidden_dim)
        embedded_sq = embedded.squeeze(1)   # (batch, embedding_dim)
        prediction  = self.fc_out(
            torch.cat([gru_out_sq, context, embedded_sq], dim=1)
        )  # (batch, vocab_size)

        return prediction, new_hidden, attn_weights


# ---------------------------------------------------------------------------
# Seq2Seq
# ---------------------------------------------------------------------------
class Seq2Seq(nn.Module):

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_pad_idx: int = 0,
        teacher_forcing_ratio: float = 0.5,
    ):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def create_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        # True where token is NOT padding
        return src != self.src_pad_idx

    def forward(
        self,
        src: torch.Tensor,              # (batch, src_len)
        tgt: torch.Tensor,              # (batch, tgt_len)
        teacher_forcing_ratio: Optional[float] = None,  # Fix #9: correct type hint
    ) -> torch.Tensor:

        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.teacher_forcing_ratio

        batch_size, tgt_len = tgt.shape
        tgt_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, tgt_len - 1, tgt_vocab_size, device=src.device)

        encoder_outputs, hidden = self.encoder(src)
        src_mask = self.create_src_mask(src)

        dec_input = tgt[:, 0]  # <sos> token

        for t in range(1, tgt_len):
            prediction, hidden, _ = self.decoder(dec_input, hidden, encoder_outputs, src_mask)
            outputs[:, t - 1, :] = prediction

            use_teacher = random.random() < teacher_forcing_ratio
            dec_input = tgt[:, t] if use_teacher else prediction.argmax(dim=1)

        return outputs


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_seq2seq(
    src_vocab_size: int,
    tgt_vocab_size: int,
    embedding_dim: int = 256,
    hidden_dim: int = 512,
    attention_dim: int = 256,
    num_layers: int = 1,
    dropout: float = 0.3,
    bidirectional_encoder: bool = True,
    src_pad_idx: int = 0,
    tgt_pad_idx: int = 0,
    teacher_forcing_ratio: float = 0.5,
) -> Seq2Seq:

    encoder_hidden_dim = hidden_dim * (2 if bidirectional_encoder else 1)

    encoder = Encoder(
        vocab_size=src_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional_encoder,
        pad_idx=src_pad_idx,
    )

    decoder = Decoder(
        vocab_size=tgt_vocab_size,
        embedding_dim=embedding_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        hidden_dim=hidden_dim,
        attention_dim=attention_dim,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=tgt_pad_idx,
    )

    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=src_pad_idx,
        teacher_forcing_ratio=teacher_forcing_ratio,
    )

    # Fix #10: use appropriate initialisers per parameter type
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue  # skip biases / 1-D params
        if "embedding" in name:
            # Embeddings: small normal distribution works better than Xavier
            nn.init.normal_(param, mean=0.0, std=0.01)
        else:
            # Linear / GRU weight matrices: Xavier uniform
            nn.init.xavier_uniform_(param)

    return model


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    SRC_VOCAB, TGT_VOCAB = 5000, 8000
    BATCH, SRC_LEN, TGT_LEN = 8, 20, 18

    # Test with multi-layer encoder/decoder (the previously broken path)
    model = build_seq2seq(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        num_layers=2,  # tests the multi-layer fix
    ).to(device)

    src = torch.randint(1, SRC_VOCAB, (BATCH, SRC_LEN)).to(device)
    tgt = torch.randint(1, TGT_VOCAB, (BATCH, TGT_LEN)).to(device)

    with torch.no_grad():
        out = model(src, tgt, teacher_forcing_ratio=0.5)

    print(f"Output shape: {out.shape}")   # expected: (8, 17, 8000)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {param_count:,}")
    print("Seq2Seq — OK")
