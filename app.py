
import os, sys, time, logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ml.classifier import IntentClassifier, build_conversational_reply, train_and_save, LABEL_NAMES
from ml.deep_model import DeepIntentTrainer

#  CLI arguments 
import argparse as _argparse
_parser = _argparse.ArgumentParser(description="IntentLens Flask App")
_parser.add_argument("--transformer", action="store_true",
                     help="Use HuggingFace DistilBERT instead of custom Transformer")
_parser.add_argument("--port", type=int, default=5000, help="Port to run on")
_args, _ = _parser.parse_known_args()
USE_DISTILBERT = _args.transformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

clf = IntentClassifier()         # LR baseline (fallback)
deep_trainer = None              # Transformer deep model (primary)
_train_metrics = {}
_deep_metrics  = {}

def init_model():
    global clf, deep_trainer, _train_metrics, _deep_metrics

    #  Load / train deep transformer model (primary) 
    global deep_trainer
    if USE_DISTILBERT:
        log.info("Using HuggingFace DistilBERT (--transformer flag set)…")
        try:
            from ml.distilbert_model import DistilBERTTrainer
            deep_trainer = DistilBERTTrainer(epochs=5, batch_size=16, lr=2e-5)
        except ImportError:
            log.error("transformers library not installed. Run: pip install transformers")
            log.error("Falling back to custom Transformer.")
            deep_trainer = DeepIntentTrainer(
                vocab_size=10000, embed_dim=128, num_heads=4,
                num_layers=3, ffn_dim=256, max_len=64,
                batch_size=128, epochs=8, lr=5e-4, warmup_steps=100
            )
    else:
        deep_trainer = DeepIntentTrainer(
            vocab_size=10000, embed_dim=128, num_heads=4,
            num_layers=3, ffn_dim=256, max_len=64,
            batch_size=128, epochs=8, lr=5e-4, warmup_steps=100
        )

    if deep_trainer.model_exists:
        log.info("Loading deep transformer model…")
        deep_trainer.load()
        log.info("Deep model ready..")
        #  Evaluate on held-out test set so sidebar shows real metrics 
        try:
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from pathlib import Path as _Pth
            _ds = _Pth("data/dataset.csv")
            if _ds.exists():
                _df = pd.read_csv(_ds)
                _, _te = train_test_split(_df, test_size=0.2,
                                          stratify=_df["label"], random_state=42)
                _, _te = train_test_split(_te, test_size=0.5,
                                          stratify=_te["label"], random_state=42)
                _deep_metrics = deep_trainer.evaluate(
                    _te["text"].tolist(), _te["label"].tolist(), split_name="Loaded"
                )
                log.info(f"Metrics: acc={_deep_metrics['accuracy']:.4f}  f1={_deep_metrics['f1']:.4f}")
        except Exception as _e:
            log.warning(f"Could not evaluate on load: {_e}")
    else:
        log.info("Training deep transformer model…")
        import pandas as pd
        from pathlib import Path
        from sklearn.model_selection import train_test_split
        from data.dataset_builder import build_dataset_v2

        # Auto-generate dataset if it doesn't exist
        dataset_path = Path("data/dataset.csv")
        if not dataset_path.exists():
            log.info("dataset.csv not found — generating now…")
            build_dataset_v2(str(dataset_path))

        df = pd.read_csv(dataset_path)
        tr, te = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
        va, te = train_test_split(te, test_size=0.5, stratify=te["label"], random_state=42)
        deep_trainer.fit(tr["text"].tolist(), tr["label"].tolist(),
                         va["text"].tolist(), va["label"].tolist())
        _deep_metrics = deep_trainer.evaluate(te["text"].tolist(), te["label"].tolist())
        deep_trainer.save()

    #  Load / train LR model (baseline / fallback) 
    if clf.model_exists:
        clf.load()
    else:
        log.info("Training LR baseline…")
        from pathlib import Path as _P
        if not _P("data/dataset.csv").exists():
            from data.dataset_builder import build_dataset_v2 as _bdv2
            _bdv2("data/dataset.csv")
        _train_metrics = train_and_save()
        clf.load()
    log.info("All models ready ✓")

with app.app_context():
    init_model()

_start_time = time.time()

# Each session key → list of past turns (label_id, label, text)
# For simplicity we keep one global session; in production use Flask sessions or a DB
_global_history: list[dict] = []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": clf.trained,
                    "uptime_s": round(time.time() - _start_time, 1),
                    "requests_served": len(_global_history)})


@app.route("/api/model/info")
def model_info():
    label_counts = {}
    for item in _global_history:
        l = item.get("label", "")
        label_counts[l] = label_counts.get(l, 0) + 1
    dm = _deep_metrics or {}
    return jsonify({
        "model": "Transformer Deep Model (3-layer, 4-head attention) + LR baseline",
        "primary": "Transformer",
        "labels": LABEL_NAMES,
        "accuracy": dm.get("accuracy") or _train_metrics.get("accuracy"),
        "f1":       dm.get("f1")       or _train_metrics.get("f1"),
        "label_distribution": label_counts,
        "total_analyzed": len(_global_history),
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text    = (data.get("text") or "").strip()
    history = data.get("history") or []   # client sends its own conversation history

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > 1000:
        return jsonify({"error": "Text too long (max 1000 chars)"}), 400
    if not clf.trained:
        return jsonify({"error": "Model not loaded"}), 503

    t0 = time.time()
    # Use deep transformer as primary model, fall back to LR if needed
    active_clf = deep_trainer if (deep_trainer and deep_trainer.model_exists) else clf
    result = build_conversational_reply(text, active_clf, history)
    result["model_name"] = ("Transformer (4-layer, multi-head attention)"
                            if active_clf is deep_trainer else "TF-IDF + Logistic Regression")
    ms     = round((time.time() - t0) * 1000, 1)
    result["inference_ms"] = ms

    # save to global history
    _global_history.append({"text": text[:80], "label": result["label"],
                             "label_id": result["label_id"],
                             "confidence": result["confidence"]})
    if len(_global_history) > 500:
        _global_history.pop(0)

    log.info(f"[analyze] '{text[:50]}' → {result['label']} ({result['confidence']}%) "
             f"ctx={result['context_used']} {ms}ms")
    return jsonify(result)


@app.route("/api/reset", methods=["POST"])
def reset():
    _global_history.clear()
    return jsonify({"status": "ok"})


@app.route("/api/train", methods=["POST"])
def retrain():
    global clf, _train_metrics
    _train_metrics = train_and_save()
    clf.load()
    return jsonify({"status": "ok", "metrics": {
        "accuracy": round(_train_metrics["accuracy"], 4),
        "f1": round(_train_metrics["f1"], 4),
    }})


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    port  = _args.port or int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    log.info(f"Starting IntentLens on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
