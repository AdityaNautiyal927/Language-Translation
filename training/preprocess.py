
from __future__ import annotations

import os
import re
import pickle
from collections import Counter
from typing import List, Dict, Optional



PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class Vocabulary:
   

    def __init__(self, lang: str):
        self.lang: str = lang
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.freq: Counter = Counter()
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.word2idx[token] = idx
            self.idx2word[idx]   = token

    def build(
        self,
        sentences: List[str],
        tokenizer: "Tokenizer",
        min_freq: int = 2,
        max_vocab_size: Optional[int] = None,
    ) -> "Vocabulary":
    
        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence, self.lang)
            self.freq.update(tokens)

        # Sort by frequency descending then alphabetically for determinism
        sorted_words = sorted(self.freq.items(), key=lambda x: (-x[1], x[0]))

        for word, count in sorted_words:
            if count < min_freq:
                continue
            if max_vocab_size and len(self.word2idx) >= max_vocab_size:
                break
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx]  = word

        return self


    def numericalize(self, sentence: str, tokenizer: "Tokenizer") -> List[int]:
       
        tokens = tokenizer.tokenize(sentence, self.lang)
        return [self.word2idx.get(t, UNK_IDX) for t in tokens]

    def denumericalize(self, indices: List[int]) -> str:
        
        stop = {PAD_IDX, SOS_IDX, EOS_IDX}
        tokens = [
            self.idx2word.get(i, UNK_TOKEN)
            for i in indices
            if i not in stop
        ]
        return " ".join(tokens)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.word2idx)

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    def __repr__(self) -> str:
        return f"Vocabulary(lang='{self.lang}', size={len(self)})"

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist vocabulary to disk as a pickle file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Vocabulary":
        """Load a previously saved vocabulary from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


class Tokenizer:
    

    _strategies: Dict[str, callable] = {}  # lang_code → callable

    def tokenize(self, text: str, lang: str) -> List[str]:
        
        strategy = self._strategies.get(lang, self._default_tokenize)
        return strategy(text)

    @staticmethod
    def _default_tokenize(text: str) -> List[str]:
        
        # Lower-case (safe for Latin; no-op for Indic scripts)
        text = text.lower()
        # Insert spaces around standalone punctuation marks
        text = re.sub(r"([।॥!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])", r" \1 ", text)
        # Collapse multiple spaces
        tokens = text.split()
        return [t for t in tokens if t]

    @classmethod
    def register_strategy(cls, lang: str, fn: callable) -> None:
        
        cls._strategies[lang] = fn


def vocab_path(vocab_dir: str, lang: str) -> str:

    return os.path.join(vocab_dir, f"vocab_{lang}.pkl")


def build_and_save_vocab(
    sentences: List[str],
    lang: str,
    tokenizer: Tokenizer,
    vocab_dir: str,
    min_freq: int = 2,
    max_vocab_size: Optional[int] = None,
) -> Vocabulary:
    
    vocab = Vocabulary(lang)
    vocab.build(sentences, tokenizer, min_freq=min_freq, max_vocab_size=max_vocab_size)
    path = vocab_path(vocab_dir, lang)
    vocab.save(path)
    print(f"[preprocess] Saved |{lang}| vocab → {path}  (size={len(vocab)})")
    return vocab


def load_vocab(lang: str, vocab_dir: str) -> Vocabulary:
    """Load a persisted vocabulary for a language."""
    path = vocab_path(vocab_dir, lang)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Vocabulary for language '{lang}' not found at '{path}'.\n"
            f"Run training first:  python training/train.py --source_lang ... --target_lang ..."
        )
    return Vocabulary.load(path)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    tokenizer = Tokenizer()

    en_sentences = [
        "Hello world",
        "Good morning",
        "How are you",
        "I am fine thank you",
        "Hello again",
    ]
    hi_sentences = [
        "नमस्ते दुनिया",
        "सुप्रभात",
        "आप कैसे हैं",
        "मैं ठीक हूँ धन्यवाद",
        "फिर नमस्ते",
    ]

    vocab_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "saved_models", "vocabs",
    )

    vocab_en = build_and_save_vocab(en_sentences, "en", tokenizer, vocab_dir, min_freq=1)
    vocab_hi = build_and_save_vocab(hi_sentences, "hi", tokenizer, vocab_dir, min_freq=1)

    print(f"\nVocab EN: {vocab_en}")
    print(f"Vocab HI: {vocab_hi}")

    sample = "How are you"
    ids = vocab_en.numericalize(sample, tokenizer)
    print(f"\nNumericalized '{sample}': {ids}")
    print(f"Denomericalized:        '{vocab_en.denumericalize(ids)}'")

    # Reload from disk
    loaded = load_vocab("en", vocab_dir)
    print(f"\nReloaded from disk: {loaded}")
    print("preprocess.py — OK")
