import re, string, json, time, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from collections import Counter
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

LABEL_NAMES  = ["Platonic", "Romantic", "Ambiguous"]
LABEL_EMOJIS = ["🤝", "💕", "🤔"]
LABEL_COLORS = ["#60a5fa", "#f472b6", "#fbbf24"]
MODEL_DIR    = Path(__file__).parent.parent / "models" / "deep"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TOKENIZER
class Vocabulary:
    PAD, UNK, CLS = 0, 1, 2

    def __init__(self, max_vocab=12000):
        self.max_vocab = max_vocab
        self.w2i = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2}
        self.i2w = {0: "<PAD>", 1: "<UNK>", 2: "<CLS>"}

    @staticmethod
    def _clean(text):
        text = str(text).lower().strip()
        # Keep contractions and basic punctuation as tokens
        text = re.sub(r"([!?.,])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def build(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(self._clean(t).split())
        for word, _ in counter.most_common(self.max_vocab - 3):
            idx = len(self.w2i)
            self.w2i[word] = idx
            self.i2w[idx]  = word
        return self

    def encode(self, text, max_len=128):
        tokens = ["<CLS>"] + self._clean(text).split()
        tokens = tokens[:max_len]
        ids    = [self.w2i.get(t, self.UNK) for t in tokens]
        ids   += [self.PAD] * (max_len - len(ids))
        return ids[:max_len]

    def __len__(self): return len(self.w2i)


# DATASET
class IntentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.x = torch.tensor([vocab.encode(t, max_len) for t in texts], dtype=torch.long)
        self.y = torch.tensor(list(labels), dtype=torch.long)
        self.mask = (self.x != 0).float()   # attention mask

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.mask[i], self.y[i]


# POSITIONAL ENCODING
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1), :])


# TRANSFORMER INTENT CLASSIFIER

class TransformerIntentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4,
                 num_layers=4, ffn_dim=256, max_len=128,
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(embed_dim, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm  = nn.LayerNorm(embed_dim)

        # Classification head
        self.drop1 = nn.Dropout(0.3)
        self.fc1   = nn.Linear(embed_dim, 64)
        self.act   = nn.GELU()
        self.drop2 = nn.Dropout(0.2)
        self.fc2   = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        # x: (B, L)
        emb = self.embed(x) * (self.embed.embedding_dim ** 0.5)
        emb = self.pos_enc(emb)                      # (B, L, D)

        # Create padding mask for transformer (True = ignore)
        key_mask = None
        if mask is not None:
            key_mask = (mask == 0)                   # (B, L)

        out  = self.transformer(emb, src_key_padding_mask=key_mask)
        out  = self.norm(out)                        # (B, L, D)

        # Mean pooling over non-padding tokens (like DistilBERT pooling)
        if mask is not None:
            m    = mask.unsqueeze(-1)                # (B, L, 1)
            pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
        else:
            pooled = out.mean(1)                     # (B, D)

        z = self.drop1(pooled)
        z = self.fc1(z)
        z = self.act(z)
        z = self.drop2(z)
        return self.fc2(z)                           # (B, num_classes)


# TRAINER
class DeepIntentTrainer:
    def __init__(self, vocab_size=12000, embed_dim=128, num_heads=4,
                 num_layers=4, ffn_dim=256, max_len=128,
                 batch_size=64, epochs=20, lr=3e-4,
                 warmup_steps=200, weight_decay=1e-4):
        self.max_len    = max_len
        self.batch_size = batch_size
        self.epochs     = epochs
        self.lr         = lr
        self.warmup     = warmup_steps
        self.vocab      = Vocabulary(vocab_size)
        self.model_cfg  = dict(embed_dim=embed_dim, num_heads=num_heads,
                               num_layers=num_layers, ffn_dim=ffn_dim,
                               max_len=max_len)
        self.model      = None
        self.history    = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

    def fit(self, train_texts, train_labels, val_texts, val_labels):
        # Build vocab on training data
        self.vocab.build(train_texts)
        vocab_size = len(self.vocab)

        self.model = TransformerIntentClassifier(
            vocab_size=vocab_size, **self.model_cfg
        ).to(DEVICE)

        total_params    = sum(p.numel() for p in self.model.parameters())
        trainable_params= sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nTransformer Classifier on {DEVICE}")
        print(f"  Vocab size : {vocab_size:,}")
        print(f"  Parameters : {total_params:,} total | {trainable_params:,} trainable")
        print(f"  Architecture: {self.model_cfg['num_layers']} layers × "
              f"{self.model_cfg['num_heads']} heads × "
              f"{self.model_cfg['embed_dim']}d | FFN={self.model_cfg['ffn_dim']}")
        print(f"  Training   : {len(train_texts)} train | {len(val_texts)} val | "
              f"{self.epochs} epochs | batch={self.batch_size} | lr={self.lr}")

        train_ds = IntentDataset(train_texts, train_labels, self.vocab, self.max_len)
        val_ds   = IntentDataset(val_texts,   val_labels,   self.vocab, self.max_len)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size)

        # Class-balanced loss
        counts = Counter(train_labels)
        weights = torch.tensor(
            [1.0 / counts[i] for i in range(3)], dtype=torch.float
        ).to(DEVICE)
        weights = weights / weights.sum() * 3
        criterion = nn.CrossEntropyLoss(weight=weights)

        optimizer = AdamW(self.model.parameters(), lr=self.lr,
                          weight_decay=self.weight_decay if hasattr(self,'weight_decay') else 1e-4)

        # Linear warmup + cosine decay
        total_steps = len(train_dl) * self.epochs
        def lr_lambda(step):
            if step < self.warmup:
                return step / max(1, self.warmup)
            progress = (step - self.warmup) / max(1, total_steps - self.warmup)
            return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_val_f1, best_state = 0.0, None
        print("─" * 70)

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            #  Train 
            self.model.train()
            t_loss, t_correct, t_total = 0.0, 0, 0
            for xb, mb, yb in train_dl:
                xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                logits = self.model(xb, mb)
                loss   = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                t_loss    += loss.item() * len(yb)
                t_correct += (logits.argmax(-1) == yb).sum().item()
                t_total   += len(yb)

            #  Validate 
            v_loss, v_acc, v_f1 = self._eval_loop(val_dl, criterion)
            tr_loss = t_loss / t_total
            tr_acc  = t_correct / t_total
            elapsed = time.time() - t0

            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(v_acc)

            # Save best model by val F1
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                best_state  = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            print(f"  Epoch {epoch:02d}/{self.epochs} | "
                  f"loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                  f"val_loss={v_loss:.4f} acc={v_acc:.3f} f1={v_f1:.3f} | "
                  f"{elapsed:.1f}s  {'★' if v_f1==best_val_f1 else ''}")

        # Restore best weights
        if best_state:
            self.model.load_state_dict(best_state)
        print(f"\nBest val F1: {best_val_f1:.4f}")
        return self.history

    def _eval_loop(self, loader, criterion):
        self.model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for xb, mb, yb in loader:
                xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
                logits = self.model(xb, mb)
                loss   = criterion(logits, yb)
                total_loss += loss.item() * len(yb)
                all_preds.extend(logits.argmax(-1).cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds, average="macro")
        return total_loss / len(all_labels), acc, f1

    def predict_proba(self, texts):
        self.model.eval()
        max_len = self.model_cfg.get("max_len", self.max_len)
        ds = IntentDataset(texts, [0]*len(texts), self.vocab, max_len)
        dl = DataLoader(ds, batch_size=64)
        probs = []
        with torch.no_grad():
            for xb, mb, _ in dl:
                xb, mb = xb.to(DEVICE), mb.to(DEVICE)
                logits = self.model(xb, mb)
                probs.append(torch.softmax(logits, -1).cpu().numpy())
        return np.vstack(probs)

    def predict(self, texts):
        return self.predict_proba(texts).argmax(axis=1)

    def evaluate(self, texts, labels, split_name="Test"):
        preds = self.predict(texts)
        acc   = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
        cm    = confusion_matrix(labels, preds)
        print(f"\n{'='*60}")
        print(f"  Transformer Deep Model — {split_name}")
        print(f"{'='*60}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")
        print(f"\n{classification_report(labels, preds, target_names=LABEL_NAMES)}")
        print("Confusion Matrix:")
        print(cm)
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1,
                "confusion_matrix": cm.tolist(),
                "report": classification_report(labels, preds, target_names=LABEL_NAMES)}

    def save(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_DIR / "model.pt")
        joblib.dump(self.vocab, MODEL_DIR / "vocab.pkl")
        joblib.dump(self.model_cfg, MODEL_DIR / "config.pkl")
        joblib.dump(self.history, MODEL_DIR / "history.pkl")
        print(f"Deep model saved → {MODEL_DIR}")

    def load(self):
        self.vocab      = joblib.load(MODEL_DIR / "vocab.pkl")
        self.model_cfg  = joblib.load(MODEL_DIR / "config.pkl")
        self.model      = TransformerIntentClassifier(
            vocab_size=len(self.vocab), **self.model_cfg
        ).to(DEVICE)
        self.model.load_state_dict(
            torch.load(MODEL_DIR / "model.pt", map_location=DEVICE)
        )
        self.model.eval()
        if (MODEL_DIR / "history.pkl").exists():
            self.history = joblib.load(MODEL_DIR / "history.pkl")
        return self

    @property
    def model_exists(self):
        return (MODEL_DIR / "model.pt").exists()


# INTERFACE ADAPTER — makes DeepIntentTrainer a drop-in for IntentClassifier
import numpy as _np

def _add_context_interface(cls):
    """Adds predict_proba_text and predict_proba_context to DeepIntentTrainer."""

    def predict_proba_text(self, text: str) -> _np.ndarray:
        """Single-text probability prediction."""
        return self.predict_proba([text])[0]

    def predict_proba_context(self, text: str, history: list) -> _np.ndarray:
        """
        Context-aware prediction with adaptive blending.
        Identical logic to IntentClassifier.predict_proba_context.
        """
        current = self.predict_proba_text(text)
        if not history:
            return current

        hist_vec, weights = _np.zeros(3), 0.0
        for i, turn in enumerate(history[-5:]):
            lbl = turn.get("label_id")
            if lbl is not None:
                w = (i + 1) * 0.5
                one_hot = _np.zeros(3)
                one_hot[int(lbl)] = 1.0
                hist_vec += w * one_hot
                weights  += w

        if weights == 0:
            return current

        hist_vec /= weights
        current_top_p = float(current.max())
        hist_top_id   = int(_np.argmax(hist_vec))
        hist_top_p    = float(hist_vec[hist_top_id])

        # Strong current signal → trust it, barely use context
        if current_top_p >= 0.55:
            return current

        ctx_weight = 0.32 if (hist_top_id == 1 and hist_top_p >= 0.60) else 0.15
        blended = _np.clip((1 - ctx_weight) * current + ctx_weight * hist_vec, 1e-9, None)
        blended /= blended.sum()

        # Safety: never flatten a confident prediction
        if float(blended.max()) < current_top_p * 0.80:
            return current
        return blended

    @property
    def trained(self):
        return self.model is not None

    cls.predict_proba_text    = predict_proba_text
    cls.predict_proba_context = predict_proba_context
    cls.trained               = trained
    return cls

DeepIntentTrainer = _add_context_interface(DeepIntentTrainer)
