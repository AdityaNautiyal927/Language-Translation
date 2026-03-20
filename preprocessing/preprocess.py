
import os
import re
import json
import unicodedata
from collections import Counter
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Special tokens & indices
# ---------------------------------------------------------------------------
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns (avoid recompiling on every call) — Fix #5
# ---------------------------------------------------------------------------
_RE_EN_ALLOWED   = re.compile(r"[^a-z0-9\s\.,!?\'\\-]")
_RE_CTRL_CHARS   = re.compile(r"[\x00-\x1f\x7f]")
_RE_MULTI_SPACE  = re.compile(r"\s+")
_RE_PUNCT_SPLIT  = re.compile(r"([.,!?])")


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
class Vocabulary:
    def __init__(self, name: str, min_freq: int = 2):
        self.name = name
        self.min_freq = min_freq

        self.token2idx: Dict[str, int] = {
            PAD_TOKEN: PAD_IDX,
            SOS_TOKEN: SOS_IDX,
            EOS_TOKEN: EOS_IDX,
            UNK_TOKEN: UNK_IDX,
        }
        self.idx2token: Dict[int, str] = {v: k for k, v in self.token2idx.items()}
        self.token_freq: Counter = Counter()

    def build_vocab(self, sentences: List[List[str]]) -> None:
        for tokens in sentences:
            self.token_freq.update(tokens)

        for token, freq in self.token_freq.items():
            if freq >= self.min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

        print(f"[Vocab: {self.name}] Size: {len(self.token2idx)} tokens "
              f"(min_freq={self.min_freq})")

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token2idx.get(t, UNK_IDX) for t in tokens]

    def decode(self, indices: List[int], skip_special: bool = True) -> List[str]:
        special = {PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX} if skip_special else set()
        return [self.idx2token[i] for i in indices
                if i in self.idx2token and i not in special]

    def __len__(self) -> int:
        return len(self.token2idx)

    def save(self, path: str) -> None:
        # Fix #2: guard against empty dirname (bare filename path)
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "name": self.name,
                "min_freq": self.min_freq,
                "token2idx": self.token2idx,
            }, f, ensure_ascii=False, indent=2)
        print(f"[Vocab] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(name=data["name"], min_freq=data["min_freq"])
        vocab.token2idx = data["token2idx"]
        vocab.idx2token = {int(v): k for k, v in data["token2idx"].items()}
        print(f"[Vocab] Loaded '{vocab.name}' — {len(vocab)} tokens from {path}")
        return vocab


# ---------------------------------------------------------------------------
# Text normalisation helpers (use pre-compiled patterns)
# ---------------------------------------------------------------------------
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_english(text: str) -> str:
    text = normalize_unicode(text.lower().strip())
    text = _RE_EN_ALLOWED.sub("", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    return text


def normalize_indic(text: str) -> str:
    text = normalize_unicode(text.strip())
    text = _RE_CTRL_CHARS.sub("", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    return text


def tokenize_en(text: str) -> List[str]:
    text = normalize_english(text)
    text = _RE_PUNCT_SPLIT.sub(r" \1 ", text)
    return text.split()


def tokenize_indic(text: str) -> List[str]:
    text = normalize_indic(text)
    return text.split()


# ---------------------------------------------------------------------------
# Corpus reader
# ---------------------------------------------------------------------------
def read_parallel_corpus(
    src_path: str,
    tgt_path: str,
    max_len: int = 50,
    max_samples: Optional[int] = None,
) -> Tuple[List[List[str]], List[List[str]]]:
    src_sentences, tgt_sentences = [], []

    with open(src_path, "r", encoding="utf-8") as sf, \
         open(tgt_path, "r", encoding="utf-8") as tf:

        for i, (src_line, tgt_line) in enumerate(zip(sf, tf)):
            if max_samples and i >= max_samples:
                break

            src_tokens = tokenize_en(src_line)
            tgt_tokens = tokenize_indic(tgt_line)

            if not src_tokens or not tgt_tokens:
                continue
            if len(src_tokens) > max_len or len(tgt_tokens) > max_len:
                continue

            src_sentences.append(src_tokens)
            tgt_sentences.append(tgt_tokens)

    print(f"[Corpus] Loaded {len(src_sentences)} sentence pairs "
          f"(max_len={max_len})")
    return src_sentences, tgt_sentences


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TranslationDataset(Dataset):

    def __init__(
        self,
        src_sentences: List[List[str]],
        tgt_sentences: List[List[str]],
        src_vocab: "Vocabulary",
        tgt_vocab: "Vocabulary",
    ):
        # Fix #1: assert now has a proper error message
        assert len(src_sentences) == len(tgt_sentences), (
            f"Source and target sentence counts must match: "
            f"{len(src_sentences)} vs {len(tgt_sentences)}"
        )
        self.src_data = src_sentences
        self.tgt_data = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        src_encoded = [SOS_IDX] + self.src_vocab.encode(self.src_data[idx]) + [EOS_IDX]
        tgt_encoded = [SOS_IDX] + self.tgt_vocab.encode(self.tgt_data[idx]) + [EOS_IDX]
        return src_encoded, tgt_encoded


# ---------------------------------------------------------------------------
# Collation — Fix #6: use pad_sequence instead of manual list comprehension
# ---------------------------------------------------------------------------
def collate_fn(
    batch: List[Tuple[List[int], List[int]]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    src_batch, tgt_batch = zip(*batch)

    # Convert to tensors first, then use optimised pad_sequence
    src_tensors = [torch.tensor(s, dtype=torch.long) for s in src_batch]
    tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in tgt_batch]

    # batch_first=True, padding_value=PAD_IDX
    src_padded = rnn_utils.pad_sequence(src_tensors, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = rnn_utils.pad_sequence(tgt_tensors, batch_first=True, padding_value=PAD_IDX)

    return src_padded, tgt_padded


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def get_dataloader(
    dataset: TranslationDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),  # speeds up GPU transfers when GPU is present
    )


# ---------------------------------------------------------------------------
# Full pipeline builder
# ---------------------------------------------------------------------------
def build_pipeline(
    src_path: str,
    tgt_path: str,
    src_vocab_path: str,
    tgt_vocab_path: str,
    max_len: int = 50,
    min_freq: int = 2,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    reload_vocab: bool = False,
) -> Tuple[DataLoader, "Vocabulary", "Vocabulary"]:

    src_sentences, tgt_sentences = read_parallel_corpus(
        src_path, tgt_path, max_len=max_len, max_samples=max_samples
    )

    # Fix #3: load cached vocab when reload_vocab=False (was inverted before)
    if not reload_vocab and os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
        src_vocab = Vocabulary.load(src_vocab_path)
        tgt_vocab = Vocabulary.load(tgt_vocab_path)
    else:
        src_vocab = Vocabulary(name="english", min_freq=min_freq)
        tgt_vocab = Vocabulary(name="target", min_freq=min_freq)
        src_vocab.build_vocab(src_sentences)
        tgt_vocab.build_vocab(tgt_sentences)
        src_vocab.save(src_vocab_path)
        tgt_vocab.save(tgt_vocab_path)

    dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
    dataloader = get_dataloader(dataset, batch_size=batch_size)

    return dataloader, src_vocab, tgt_vocab


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dummy_src = [["hello", "world"], ["how", "are", "you"], ["good", "morning"]]
    dummy_tgt = [["नमस्ते", "दुनिया"], ["आप", "कैसे", "हैं"], ["सुप्रभात"]]

    src_vocab = Vocabulary("english", min_freq=1)
    tgt_vocab = Vocabulary("hindi", min_freq=1)
    src_vocab.build_vocab(dummy_src)
    tgt_vocab.build_vocab(dummy_tgt)

    dataset = TranslationDataset(dummy_src, dummy_tgt, src_vocab, tgt_vocab)
    loader = get_dataloader(dataset, batch_size=2, shuffle=False)

    for src_batch, tgt_batch in loader:
        print("SRC batch shape:", src_batch.shape)
        print("TGT batch shape:", tgt_batch.shape)
        print("Sample SRC decoded:", src_vocab.decode(src_batch[0].tolist()))
        print("Sample TGT decoded:", tgt_vocab.decode(tgt_batch[0].tolist()))
        break

    print("\n Preprocessing pipeline test passed")
