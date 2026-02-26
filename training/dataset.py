

from __future__ import annotations

import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple

# Allow running as a script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.preprocess import Vocabulary, Tokenizer, SOS_IDX, EOS_IDX, PAD_IDX


class TranslationDataset(Dataset):
    

    def __init__(
        self,
        csv_path: str,
        src_lang: str,
        tgt_lang: str,
        vocab_src: Vocabulary,
        vocab_tgt: Vocabulary,
        tokenizer: Tokenizer,
        max_src_len: int = 150,
        max_tgt_len: int = 150,
    ):
        self.vocab_src  = vocab_src
        self.vocab_tgt  = vocab_tgt
        self.tokenizer  = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        df = pd.read_csv(csv_path, dtype=str).dropna()

        # Filter to the requested language pair
        mask = (df["source_lang"].str.strip() == src_lang) & \
               (df["target_lang"].str.strip() == tgt_lang)
        df = df[mask].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"No rows found for language pair {src_lang}→{tgt_lang} "
                f"in '{csv_path}'. Check 'source_lang' / 'target_lang' columns."
            )

        self.src_sentences: List[str] = df["source_text"].tolist()
        self.tgt_sentences: List[str] = df["target_text"].tolist()

        print(
            f"[dataset] Loaded {len(self.src_sentences)} pairs "
            f"({src_lang}→{tgt_lang}) from '{csv_path}'"
        )

    def __len__(self) -> int:
        return len(self.src_sentences)

    def __getitem__(self, idx: int) -> Dict:
        src_text = self.src_sentences[idx]
        tgt_text = self.tgt_sentences[idx]

        src_ids = self._encode(src_text, self.vocab_src, self.max_src_len)
        tgt_ids = self._encode(tgt_text, self.vocab_tgt, self.max_tgt_len)

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "src_len": len(src_ids),
            "tgt_len": len(tgt_ids),
        }

    def _encode(self, text: str, vocab: Vocabulary, max_len: int) -> List[int]:
        """Tokenise → numericalize → wrap with <SOS>/<EOS> → truncate."""
        token_ids = vocab.numericalize(text, self.tokenizer)
        token_ids = token_ids[:max_len]  # truncate before adding specials
        return [SOS_IDX] + token_ids + [EOS_IDX]


    def split(self, val_fraction: float = 0.1, seed: int = 42):
        
        import math, random as rng
        indices = list(range(len(self)))
        rng.seed(seed)
        rng.shuffle(indices)
        n_val = max(1, math.floor(len(indices) * val_fraction))
        val_idx   = indices[:n_val]
        train_idx = indices[n_val:]
        return _SubsetDataset(self, train_idx), _SubsetDataset(self, val_idx)


class _SubsetDataset(Dataset):

    def __init__(self, base: TranslationDataset, indices: List[int]):
        self.base    = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        return self.base[self.indices[idx]]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    
    src_list  = [item["src_ids"] for item in batch]
    tgt_list  = [item["tgt_ids"] for item in batch]
    src_lens  = torch.tensor([item["src_len"] for item in batch])
    tgt_lens  = torch.tensor([item["tgt_len"] for item in batch])

    # pad_sequence expects (T, batch) → we swap to (batch, T) after
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=PAD_IDX)

    return {
        "src"     : src_padded,
        "tgt"     : tgt_padded,
        "src_lens": src_lens,
        "tgt_lens": tgt_lens,
    }


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
   
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_vocabs_from_csv(
    csv_path: str,
    src_lang: str,
    tgt_lang: str,
    tokenizer: Tokenizer,
    vocab_dir: str,
    min_freq: int = 1,
    max_vocab_size: int = None,
) -> Tuple[Vocabulary, Vocabulary]:
   
    from training.preprocess import build_and_save_vocab

    df = pd.read_csv(csv_path, dtype=str).dropna()
    mask = (df["source_lang"].str.strip() == src_lang) & \
           (df["target_lang"].str.strip() == tgt_lang)
    df = df[mask]

    src_sentences = df["source_text"].tolist()
    tgt_sentences = df["target_text"].tolist()

    vocab_src = build_and_save_vocab(
        src_sentences, src_lang, tokenizer, vocab_dir,
        min_freq=min_freq, max_vocab_size=max_vocab_size,
    )
    vocab_tgt = build_and_save_vocab(
        tgt_sentences, tgt_lang, tokenizer, vocab_dir,
        min_freq=min_freq, max_vocab_size=max_vocab_size,
    )
    return vocab_src, vocab_tgt


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV_PATH     = os.path.join(PROJECT_ROOT, "data", "en_hi.csv")
    VOCAB_DIR    = os.path.join(PROJECT_ROOT, "saved_models", "vocabs")

    tokenizer = Tokenizer()

    vocab_src, vocab_tgt = build_vocabs_from_csv(
        CSV_PATH, "en", "hi", tokenizer, VOCAB_DIR, min_freq=1
    )

    dataset = TranslationDataset(
        CSV_PATH, "en", "hi", vocab_src, vocab_tgt, tokenizer
    )
    train_ds, val_ds = dataset.split(val_fraction=0.1)

    train_loader = get_dataloader(train_ds, batch_size=4, shuffle=True)
    batch = next(iter(train_loader))

    print(f"\nsrc shape : {batch['src'].shape}")   # (4, max_src_len)
    print(f"tgt shape : {batch['tgt'].shape}")   # (4, max_tgt_len)
    print(f"src_lens  : {batch['src_lens']}")
    print(f"tgt_lens  : {batch['tgt_lens']}")
    print("dataset.py — OK")
