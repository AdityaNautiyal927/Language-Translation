# Multilingual Neural Machine Translation System

A scalable, modular **Seq2Seq + Bahdanau Attention** NMT system built with PyTorch and served via FastAPI.  
Initially supports **English → Hindi** — designed to add any Indian regional language with a single new dataset file and one training command.

---

## 📁 Project Structure

```
Language_Translation/
├── config/
│   ├── languages.yaml          # Supported languages registry
│   └── training_config.yaml    # All hyperparameters
├── data/
│   ├── en_hi.csv               # English–Hindi parallel corpus
│   └── en_mr.csv               # (Future) English–Marathi
├── models/
│   ├── attention.py            # BahdanauAttention module
│   └── seq2seq.py              # Encoder, Decoder, Seq2Seq, build_seq2seq()
├── training/
│   ├── preprocess.py           # Vocabulary + Tokenizer classes
│   ├── dataset.py              # TranslationDataset, collate_fn, DataLoaders
│   └── train.py                # CLI training entry point
├── inference/
│   └── translator.py           # ModelRegistry (singleton), Translator
├── app/
│   └── main.py                 # FastAPI application
├── saved_models/               # Checkpoint + vocabulary + eval outputs
│   ├── model_en_hi.pt
│   ├── vocabs/
│   │   ├── vocab_en.pkl
│   │   └── vocab_hi.pkl
│   └── evals/
│       └── eval_en_hi.json
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train English → Hindi

```bash
python training/train.py --source_lang en --target_lang hi
```

This will:
- Build `saved_models/vocabs/vocab_en.pkl` and `vocab_hi.pkl`
- Train and save `saved_models/model_en_hi.pt`  
- Write BLEU scores to `saved_models/evals/eval_en_hi.json`

### 3. Run the API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Translate a Sentence

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "source_lang": "en", "target_lang": "hi"}'
```

**Response:**
```json
{
  "translation": "नमस्ते दुनिया",
  "source_lang": "en",
  "target_lang": "hi"
}
```

---

## 🌐 API Reference

| Method | Endpoint       | Description                            |
|--------|----------------|----------------------------------------|
| POST   | `/translate`   | Translate text between language pairs  |
| GET    | `/languages`   | List all supported language codes      |
| GET    | `/models`      | List all available trained models      |
| GET    | `/health`      | Health check / readiness probe         |
| GET    | `/docs`        | Swagger UI (interactive API explorer)  |

---

## 🗣️ Adding a New Regional Language (e.g., Marathi)

### Step 1 — Prepare the dataset

Create `data/en_mr.csv` with columns:

```
source_text,target_text,source_lang,target_lang
Hello,नमस्कार,en,mr
Good morning,सुप्रभात,en,mr
...
```

### Step 2 — Register the language (optional metadata)

Open `config/languages.yaml` and uncomment / add:

```yaml
mr:
  name: Marathi
  script: Devanagari
  tokenizer: whitespace
```

### Step 3 — Train

```bash
python training/train.py \
  --source_lang en \
  --target_lang mr \
  --data_path data/en_mr.csv
```

This creates:
- `saved_models/vocabs/vocab_mr.pkl`
- `saved_models/model_en_mr.pt`
- `saved_models/evals/eval_en_mr.json`

### Step 4 — The API is automatically ready

No restart needed if you launched with `--reload`.  The new model is lazy-loaded on first request:

```bash
curl -X POST http://localhost:8000/translate \
  -d '{"text": "Hello", "source_lang": "en", "target_lang": "mr"}'
```

That is all — **zero code changes required**.

---

## 🏋️ Training Options

```
python training/train.py [options]

Required:
  --source_lang   en          Source language code
  --target_lang   hi          Target language code

Optional:
  --data_path     data/en_hi.csv  Path to bilingual CSV
  --config        config/training_config.yaml
  --save_dir      saved_models
  --num_epochs    20              Override config epochs
  --batch_size    64              Override config batch size
  --rebuild_vocab                 Force vocab rebuild
```

---

## ⚙️ Configuration

`config/training_config.yaml`:

```yaml
model:
  embedding_dim: 256
  hidden_dim: 512
  dropout: 0.3
  num_layers: 1
  bidirectional_encoder: true

training:
  batch_size: 64
  learning_rate: 0.001
  num_epochs: 20
  teacher_forcing_ratio: 0.5
  clip_grad_norm: 1.0
  val_split: 0.1
```

---

## 📊 Evaluation

BLEU scores are logged per epoch during training and saved to:

```
saved_models/evals/eval_{src}_{tgt}.json
```

Example:

```json
{
  "src_lang": "en",
  "tgt_lang": "hi",
  "best_val_loss": 3.1245,
  "best_bleu": 12.5,
  "epochs": [
    {"epoch": 1, "train_loss": 4.82, "val_loss": 4.21, "bleu": 2.1},
    {"epoch": 2, "train_loss": 3.94, "val_loss": 3.67, "bleu": 5.4}
  ]
}
```

---

## 🚀 Deployment

### Docker (recommended for production)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t nmt-api .
docker run -p 8000:8000 -v $(pwd)/saved_models:/app/saved_models nmt-api
```

### GPU Support

Training automatically uses CUDA if available.  
The inference server also uses GPU if one is present.  
No code changes required — the device is auto-detected via:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 🔧 Extending the Tokenizer

For Indic-NLP-based tokenization (better accuracy for Indian languages):

```python
# in training/preprocess.py or your setup script
from training.preprocess import Tokenizer

def tokenize_hi(text):
    from indicnlp.tokenize import indic_tokenize
    return indic_tokenize.trivial_tokenize(text)

Tokenizer.register_strategy("hi", tokenize_hi)
```

---

## 📦 Dataset Format

All datasets use a single unified CSV format:

| Column        | Type   | Example         |
|---------------|--------|-----------------|
| `source_text` | string | `Hello world`   |
| `target_text` | string | `नमस्ते दुनिया` |
| `source_lang` | string | `en`            |
| `target_lang` | string | `hi`            |

Multiple language pairs can coexist in one CSV file — the training script filters by `source_lang` / `target_lang`.

---

## 🛠️ Architecture Summary

```
Input Sentence
      │
      ▼
 [Tokenizer] ──► [Vocabulary] ──► token IDs
      │
      ▼
 [Encoder]   ── Bidirectional GRU
      │  encoder_outputs (all steps)
      │  final hidden state
      ▼
 [Decoder]   ── At each step:
      │         1. Embed previous token
      │         2. BahdanauAttention(hidden, encoder_outputs)
      │         3. GRU([embed; context])
      │         4. Linear → vocab logits
      ▼
 Greedy argmax ──► [Vocabulary.denumericalize] ──► Translation
```

---

## 📋 Supported Languages

| Code | Language | Script     |
|------|----------|------------|
| en   | English  | Latin      |
| hi   | Hindi    | Devanagari |
| mr   | Marathi  | Devanagari |
| ta   | Tamil    | Tamil      |
| bn   | Bengali  | Bengali    |
| pa   | Punjabi  | Gurmukhi   |
| te   | Telugu   | Telugu     |

> **Note:** Only `en↔hi` is shipped with sample data. Other languages require sourcing a parallel corpus and running one training command.

---

## 📚 References

- [Bahdanau et al., 2015 — "Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/abs/1409.0473)
- [Sutskever et al., 2014 — "Sequence to Sequence Learning with Neural Networks"](https://arxiv.org/abs/1409.3215)
- [IIT Bombay English-Hindi Parallel Corpus](http://www.cfilt.iitb.ac.in/iitb_parallel/) (recommended training data)
- [IndicNLP Library](https://anoopkunchukuttan.github.io/indic_nlp_library/) (for better Indic tokenization)
