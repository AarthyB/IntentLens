<div align="center">

# 🔍 IntentLens

**Deep learning system that classifies conversational text as Romantic 💕, Platonic 🤝, or Ambiguous 🤔**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21F?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=flat&logo=render&logoColor=white)](https://intentlens.onrender.com)

🔗 **[Live Demo](https://intentlens.onrender.com)** · **[Repository](https://github.com/AarthyB/IntentLens)**

</div>

---

## 📖 About

Conversational text often carries implied meaning beyond its literal words. The same message can signal romantic interest, platonic friendship, or stay genuinely ambiguous — and even humans frequently disagree.

**IntentLens** tackles this problem with a full-stack deep learning pipeline that:
- Classifies input text into **Platonic**, **Romantic**, or **Ambiguous** with confidence scores
- Compares **4 models**: Logistic Regression, Bi-LSTM, Custom Transformer, and DistilBERT
- Serves a live **context-aware chat interface** with conversation history blending
- Auto-generates a **13-file evaluation report** covering all proposal metrics

---

## 🏆 Results

| Metric | **Transformer** | LR | LSTM |
|---|---|---|---|
| Accuracy | **0.981** | 0.974 | 0.964 |
| F1 (macro) | **0.981** | 0.974 | 0.964 |
| Ambiguous F1 | **0.988** | 0.963 | 0.975 |
| Romantic ↔ Ambiguous errors | **5** | 9 | 9 |

**5-Fold CV (LR):** 0.9795 ± 0.0061

Key findings:
- **Transformer > LR > LSTM** across all metrics — confirmed proposal hypothesis
- **Romantic ↔ Ambiguous** is the dominant error mode across all models — as predicted
- Context-aware inference (conversation history blending) improves multi-turn accuracy

---

## 🏗️ Model Architecture

### Primary — Custom Transformer
```
text → tokenizer + <CLS>
     → embedding (128d) + sinusoidal positional encoding
     → 3× TransformerEncoderLayer (4 heads, 256 FFN, GELU, dropout=0.2)
     → LayerNorm → mean pooling
     → Dropout(0.3) → Linear(128→64) → GELU → Linear(64→3) → softmax
     → (label, confidence %, probability distribution)
```

### All Models

| Model | Description | Training Time |
|---|---|---|
| **Custom Transformer** | 3-layer encoder, 4 attention heads, 128d embeddings | ~3 min (CPU) |
| **DistilBERT** | `distilbert-base-uncased` fine-tuned (66M params) | Optional |
| **Bi-LSTM** | 2-layer bidirectional, 128 hidden units | ~5 min (CPU) |
| **Logistic Regression** | TF-IDF baseline, unigrams–trigrams | < 5 seconds |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/AarthyB/IntentLens.git
cd IntentLens
pip install -r requirements.txt
```

### 2. Run the web app

```bash
python app.py
```

Open **http://localhost:5000** — on first run, the dataset is auto-generated and the model trains (~3 min). Subsequent runs load saved weights instantly from `models/`.

### 3. Optional: Run with DistilBERT

```bash
pip install transformers
python app.py --transformer
```

### 4. Full 3-model evaluation

```bash
python evaluate_models.py
# Saves 13 output files to results/

python evaluate_models.py --skip-lstm      # Faster: LR + Transformer only
python evaluate_models.py --output my_dir  # Custom output folder
```

### 5. Compare Custom Transformer vs DistilBERT

```bash
python compare_transformers.py
```

---

## 📁 File Structure

```
IntentLens/
├── app.py                      # Flask server — main entry point
├── evaluate_models.py          # Full 3-model comparison & evaluation
├── compare_transformers.py     # Custom Transformer vs DistilBERT
├── requirements.txt
├── render.yaml                 # Render.com deployment config
│
├── templates/index.html        # Chat UI
├── static/
│   ├── index.css               # Dark-themed UI, animations, responsive layout
│   └── index.js                # Frontend interaction & real-time predictions
│
├── ml/
│   ├── deep_model.py           # PRIMARY: Custom PyTorch Transformer
│   ├── distilbert_model.py     # OPTIONAL: HuggingFace DistilBERT fine-tuning
│   └── classifier.py           # LR baseline + input parser + reply builder
│
├── data/
│   ├── dataset_builder.py      # Synthetic dataset generation (~4,822 samples)
│   └── dataset.csv             # Auto-generated on first run
│
├── models/                     # Saved weights (auto-created on first run)
│   ├── deep/
│   │   ├── model.pt            # Transformer weights
│   │   ├── vocab.pkl           # Word vocabulary (10,000 tokens)
│   │   ├── config.pkl          # Model hyperparameters
│   │   └── history.pkl         # Training curves
│   ├── tfidf.pkl
│   └── lr.pkl
│
└── results/                    # Generated by evaluate_models.py
    ├── 00_evaluation_report.txt
    ├── 01_dataset_distribution.png
    ├── 03_confusion_matrices.png
    ├── 04_model_comparison.png
    ├── 06_training_curves.png
    ├── 08_radar_chart.png
    ├── 10_error_analysis.csv
    └── ... (13 files total)
```

---

## 🌐 API Reference

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `POST` | `/api/analyze` | Classify text with conversation history |
| `GET` | `/api/health` | Server health + uptime |
| `GET` | `/api/model/info` | Model accuracy + session stats |
| `POST` | `/api/train` | Re-train models |
| `POST` | `/api/reset` | Clear conversation history |

**Example:**
```bash
curl -X POST https://intentlens.onrender.com/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I cant stop thinking about you", "history": []}'
```

---

## ☁️ Deploy Your Own

This project includes a `render.yaml` for one-click deployment:

1. Fork this repository
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your fork — Render auto-detects `render.yaml`
4. Click **Deploy** — the model trains automatically on first boot

> **Note:** Render free tier sleeps after 15 minutes of inactivity. First visit after sleep takes ~30 seconds.

---

## ⚙️ Commands Reference

| Command | Purpose |
|---|---|
| `python app.py` | Run web app (Custom Transformer) |
| `python app.py --transformer` | Run web app (DistilBERT) |
| `python app.py --port 8080` | Run on custom port |
| `python evaluate_models.py` | Full 3-model evaluation |
| `python evaluate_models.py --skip-lstm` | Faster: LR + Transformer only |
| `python compare_transformers.py` | Custom Transformer vs DistilBERT |

---

## ⚖️ Ethical Considerations

- All training data is **synthetic** — no real personal conversations used
- Predictions are **probabilistic**, not deterministic verdicts
- Cultural and demographic variation is acknowledged but not modelled
- Intended for **research and demonstration only**

---

## 📚 References

- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL-HLT 2019.
- Sanh et al. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108.
- Ranganath et al. (2009). *It's Not You, It's Me: Detecting Flirting in Speed-Dates.* EMNLP 2009.
- Pei & Jurgens (2020). *Quantifying Intimacy in Language.* EMNLP 2020.

---

<div align="center">
  <i>Built to explore the subtle boundary between "just friends" and something more 🤔</i>
</div>
