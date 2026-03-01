# Language Translation — NMT System

A Neural Machine Translation (NMT) system using a **Seq2Seq architecture with Bahdanau Attention**, built in PyTorch.

Supports:
- 🇬🇧 **English → Hindi** (`en → hi`)
- 🇮🇳 **Hindi → English** (`hi → en`)

---

## Project Structure

```
Language_Translation/
├── models/
│   ├── attention.py         # Bahdanau Attention mechanism
│   ├── seq2seq.py           # Encoder, Decoder, Seq2Seq model
│   └── __init__.py
├── training/
│   ├── preprocess.py        # Vocabulary + Tokenizer
│   ├── dataset.py           # TranslationDataset + DataLoader
│   └── train.py             # Training loop
├── inference/
│   └── translator.py        # Load model and translate sentences
├── app/
│   └── main.py              # FastAPI REST API
├── config/
│   ├── training_config.yaml # Hyperparameters
│   └── languages.yaml       # Supported language pairs
├── data/                    # Place your CSV datasets here (not committed)
├── saved_models/            # Checkpoints saved here (not committed)
├── evaluate.py              # BLEU score evaluation
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset

Place a CSV file at `data/en_hi.csv` with these columns:

| source_lang | target_lang | source_text | target_text |
|-------------|-------------|-------------|-------------|
| en          | hi          | Hello world | नमस्ते दुनिया |
| hi          | en          | नमस्ते दुनिया | Hello world |

Recommended dataset: [IIT Bombay English-Hindi Parallel Corpus](http://www.cfilt.iitb.ac.in/iitb_parallel/)

### 3. Train the model

English → Hindi:
```bash
python -m training.train --data_path data/en_hi.csv --source_lang en --target_lang hi
```

Hindi → English:
```bash
python -m training.train --data_path data/en_hi.csv --source_lang hi --target_lang en
```

### 4. Translate a sentence

```bash
python -m inference.translator --source_lang en --target_lang hi --text "How are you?"
```

### 5. Start the API server

```bash
uvicorn app.main:app --reload
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API.

---

## Model Architecture

```
Input (English)
     │
  Embedding
     │
  Bi-GRU Encoder
     │
  Bahdanau Attention  ←──────────────────┐
     │                                   │
  GRU Decoder  ──────────────────────────┘
     │
  Linear + Softmax
     │
Output (Hindi)
```

| Component | Detail |
|-----------|--------|
| Encoder   | Bidirectional GRU |
| Attention | Bahdanau (additive) |
| Decoder   | Unidirectional GRU |
| Embedding | 256-dim |
| Hidden    | 512-dim |

---

## Evaluate (BLEU Score)

```bash
python evaluate.py --data_path data/en_hi.csv --source_lang en --target_lang hi
```

---

## Configuration

Edit `config/training_config.yaml` to change hyperparameters (batch size, learning rate, epochs, etc.).

---

## Requirements

- Python 3.9+
- PyTorch 2.1+
- See `requirements.txt` for full list
