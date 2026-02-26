

from __future__ import annotations

import os
import sys
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.seq2seq import build_seq2seq, Seq2Seq
from training.preprocess import (
    Tokenizer, Vocabulary, load_vocab, vocab_path,
    SOS_IDX, EOS_IDX, PAD_IDX,
)



class ModelNotFoundError(FileNotFoundError):

    def __init__(self, src_lang: str, tgt_lang: str, expected_path: str):
        self.src_lang       = src_lang
        self.tgt_lang       = tgt_lang
        self.expected_path  = expected_path
        super().__init__(
            f"Model for '{src_lang}→{tgt_lang}' not found.\n"
            f"Expected checkpoint: '{expected_path}'\n"
            f"Train it first:  python training/train.py "
            f"--source_lang {src_lang} --target_lang {tgt_lang}"
        )



@dataclass
class TranslationResult:
    translation    : str
    src_lang       : str
    tgt_lang       : str
    attention_weights: Optional[List[List[float]]] = field(default=None, repr=False)



class ModelRegistry:

    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(
        self,
        models_dir: str = None,
        vocab_dir: str = None,
        device: torch.device = None,
    ):
        self._models_dir = models_dir or os.path.join(PROJECT_ROOT, "saved_models")
        self._vocab_dir  = vocab_dir  or os.path.join(PROJECT_ROOT, "saved_models", "vocabs")
        self._device     = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: Dict[Tuple[str, str], Tuple[Seq2Seq, Vocabulary, Vocabulary]] = {}
        self._per_pair_lock: Dict[Tuple[str, str], threading.RLock] = {}
        self._tokenizer = Tokenizer()

    def get(self, src_lang: str, tgt_lang: str) -> Tuple[Seq2Seq, Vocabulary, Vocabulary]:
        """
        Return (model, vocab_src, vocab_tgt) for the given language pair.

        Loads from disk on first call; returns cached copy on subsequent calls.
        Raises ModelNotFoundError if the checkpoint does not exist.
        """
        key = (src_lang, tgt_lang)

        if key not in self._per_pair_lock:
            with self._lock:
                if key not in self._per_pair_lock:
                    self._per_pair_lock[key] = threading.RLock()

        with self._per_pair_lock[key]:
            if key not in self._cache:
                self._cache[key] = self._load(src_lang, tgt_lang)
        return self._cache[key]


    def _load(self, src_lang: str, tgt_lang: str) -> Tuple[Seq2Seq, Vocabulary, Vocabulary]:
        ckpt_path = self._checkpoint_path(src_lang, tgt_lang)
        if not os.path.exists(ckpt_path):
            raise ModelNotFoundError(src_lang, tgt_lang, ckpt_path)

        print(f"[registry] Loading model {src_lang}→{tgt_lang} from '{ckpt_path}' …")
        ckpt = torch.load(ckpt_path, map_location=self._device)

        hp = ckpt["hyperparams"]
        model = build_seq2seq(
            src_vocab_size=ckpt["src_vocab_size"],
            tgt_vocab_size=ckpt["tgt_vocab_size"],
            embedding_dim=hp["embedding_dim"],
            hidden_dim=hp["hidden_dim"],
            attention_dim=hp.get("embedding_dim", 256),
            num_layers=hp.get("num_layers", 1),
            dropout=hp["dropout"],
            bidirectional_encoder=hp.get("bidirectional_encoder", True),
            src_pad_idx=0,
            tgt_pad_idx=0,
            teacher_forcing_ratio=0.0,   # always greedy during inference
        ).to(self._device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        vocab_src = load_vocab(src_lang, self._vocab_dir)
        vocab_tgt = load_vocab(tgt_lang, self._vocab_dir)

        print(f"[registry] Model ready  ({src_lang}→{tgt_lang}).")
        return model, vocab_src, vocab_tgt

    def available_pairs(self) -> List[Tuple[str, str]]:
        """
        Scan the models directory and return all available (src, tgt) pairs
        based on checkpoint filenames (model_{src}_{tgt}.pt).
        """
        pairs = []
        if not os.path.isdir(self._models_dir):
            return pairs
        for fname in os.listdir(self._models_dir):
            if fname.startswith("model_") and fname.endswith(".pt"):
                stem   = fname[len("model_"):-len(".pt")]   # "en_hi"
                parts  = stem.split("_", 1)
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
        return pairs


    def _checkpoint_path(self, src_lang: str, tgt_lang: str) -> str:
        return os.path.join(self._models_dir, f"model_{src_lang}_{tgt_lang}.pt")

    def clear_cache(self) -> None:

        with self._lock:
            self._cache.clear()


class Translator:

    def __init__(
        self,
        registry: ModelRegistry = None,
        max_output_len: int = 100,
    ):
        self._registry       = registry or ModelRegistry.instance()
        self._tokenizer      = self._registry._tokenizer
        self._device         = self._registry._device
        self.max_output_len  = max_output_len

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        return_attention: bool = False,
    ) -> TranslationResult:

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        model, vocab_src, vocab_tgt = self._registry.get(src_lang, tgt_lang)

        
        token_ids = vocab_src.numericalize(text.strip(), self._tokenizer)
        src_tensor = torch.tensor(
            [SOS_IDX] + token_ids + [EOS_IDX], dtype=torch.long
        ).unsqueeze(0).to(self._device)  # (1, src_len)

        
        with torch.no_grad():
            encoder_outputs, hidden = model.encoder(src_tensor)
            src_mask = model.create_src_mask(src_tensor)

            # Greedy decode
            dec_input = torch.tensor([SOS_IDX], device=self._device)
            gen_ids   = []
            all_attn  = []

            for _ in range(self.max_output_len):
                pred, hidden, attn_weights = model.decoder(
                    dec_input, hidden, encoder_outputs, src_mask
                )
                top1 = pred.argmax(dim=1)

                if top1.item() == EOS_IDX:
                    break

                gen_ids.append(top1.item())
                if return_attention:
                    all_attn.append(attn_weights.squeeze(0).cpu().tolist())
                dec_input = top1

        translation = vocab_tgt.denumericalize(gen_ids)

        return TranslationResult(
            translation=translation,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            attention_weights=all_attn if return_attention else None,
        )

    def available_pairs(self) -> List[Tuple[str, str]]:
       
        return self._registry.available_pairs()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--text",        default="Hello world")
    parser.add_argument("--source_lang", default="en")
    parser.add_argument("--target_lang", default="hi")
    args = parser.parse_args()

    try:
        translator = Translator()
        result = translator.translate(args.text, args.source_lang, args.target_lang)
        print(f"\nSource ({args.source_lang}): {args.text}")
        print(f"Target ({args.target_lang}): {result.translation}")
    except ModelNotFoundError as e:
        print(f"\n[ERROR] {e}")
