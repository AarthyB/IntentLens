import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)
from pathlib import Path
import joblib, time

LABEL_NAMES  = ["Platonic", "Romantic", "Ambiguous"]
LABEL_EMOJIS = ["🤝", "💕", "🤔"]
LABEL_COLORS = ["#60a5fa", "#f472b6", "#fbbf24"]
MODEL_DIR    = Path(__file__).parent.parent / "models" / "distilbert"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  Dataset 
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


#  Model: DistilBERT + classification head
class DistilBERTIntentClassifier(nn.Module):
    def __init__(self, num_labels=3, dropout=0.3):
        super().__init__()
        self.bert     = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden        = self.bert.config.hidden_size   # 768
        self.drop1    = nn.Dropout(dropout)
        self.fc1      = nn.Linear(hidden, 256)
        self.act      = nn.GELU()
        self.drop2    = nn.Dropout(0.2)
        self.fc2      = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls    = out.last_hidden_state[:, 0, :]   # [CLS] token
        x      = self.drop1(cls)
        x      = self.fc1(x)
        x      = self.act(x)
        x      = self.drop2(x)
        logits = self.fc2(x)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return loss, logits


#  Trainer 
class DistilBERTTrainer:
    def __init__(self, max_len=128, batch_size=16, epochs=5,
                 lr=2e-5, warmup_ratio=0.1, weight_decay=0.01):
        self.max_len       = max_len
        self.batch_size    = batch_size
        self.epochs        = epochs
        self.lr            = lr
        self.warmup_ratio  = warmup_ratio
        self.weight_decay  = weight_decay
        self.tokenizer     = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.model         = DistilBERTIntentClassifier().to(DEVICE)
        self.history       = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
        print(f"DistilBERT ready on {DEVICE}")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable    = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Parameters: {total_params:,} total | {trainable:,} trainable")

    def fit(self, train_texts, train_labels, val_texts, val_labels):
        train_ds = IntentDataset(train_texts, train_labels, self.tokenizer, self.max_len)
        val_ds   = IntentDataset(val_texts,   val_labels,   self.tokenizer, self.max_len)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size)

        optimizer = AdamW(self.model.parameters(), lr=self.lr,
                          weight_decay=self.weight_decay, eps=1e-8)
        total_steps  = len(train_dl) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler    = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        print(f"\nTraining: {len(train_ds)} samples | "
              f"Val: {len(val_ds)} | Epochs: {self.epochs} | "
              f"Batch: {self.batch_size} | LR: {self.lr}")
        print("─" * 65)

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            #  train 
            self.model.train()
            t_loss, t_correct, t_total = 0.0, 0, 0
            for batch in train_dl:
                input_ids = batch["input_ids"].to(DEVICE)
                attn_mask = batch["attention_mask"].to(DEVICE)
                labels    = batch["labels"].to(DEVICE)
                optimizer.zero_grad()
                loss, logits = self.model(input_ids, attn_mask, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                t_loss    += loss.item() * len(labels)
                preds      = logits.argmax(dim=-1)
                t_correct += (preds == labels).sum().item()
                t_total   += len(labels)

            #  validate 
            v_loss, v_acc = self._eval_loop(val_dl)
            tr_loss = t_loss / t_total
            tr_acc  = t_correct / t_total
            elapsed = time.time() - t0

            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(v_acc)

            print(f"  Epoch {epoch:02d}/{self.epochs} | "
                  f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                  f"val_loss={v_loss:.4f} acc={v_acc:.3f} | "
                  f"{elapsed:.0f}s")

        return self.history

    def _eval_loop(self, loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attn_mask = batch["attention_mask"].to(DEVICE)
                labels    = batch["labels"].to(DEVICE)
                loss, logits = self.model(input_ids, attn_mask, labels)
                total_loss += loss.item() * len(labels)
                correct    += (logits.argmax(-1) == labels).sum().item()
                total      += len(labels)
        return total_loss / total, correct / total

    def predict_proba(self, texts, batch_size=32):
        self.model.eval()
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc   = self.tokenizer(
                batch, truncation=True, padding="max_length",
                max_length=self.max_len, return_tensors="pt"
            ).to(DEVICE)
            with torch.no_grad():
                _, logits = self.model(enc["input_ids"], enc["attention_mask"])
                probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
        return np.vstack(all_probs)

    def predict(self, texts, batch_size=32):
        return self.predict_proba(texts, batch_size).argmax(axis=1)

    def evaluate(self, texts, labels, split_name="Test"):
        preds = self.predict(texts)
        acc   = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
        cm    = confusion_matrix(labels, preds)
        print(f"\n{'='*60}")
        print(f"  DistilBERT — {split_name}")
        print(f"{'='*60}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")
        print(f"\n{classification_report(labels, preds, target_names=LABEL_NAMES)}")
        print("Confusion Matrix:")
        print(cm)
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1,
                "confusion_matrix": cm.tolist(), "report": classification_report(
                    labels, preds, target_names=LABEL_NAMES)}

    def save(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_DIR / "model.pt")
        self.tokenizer.save_pretrained(str(MODEL_DIR))
        joblib.dump(self.history, MODEL_DIR / "history.pkl")
        print(f"DistilBERT saved → {MODEL_DIR}")

    def load(self):
        self.model.load_state_dict(
            torch.load(MODEL_DIR / "model.pt", map_location=DEVICE, weights_only=False)
        )
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_DIR))
        self.model.eval()
        return self

    @property
    def model_exists(self):
        return (MODEL_DIR / "model.pt").exists()

    #  Interface adapter: makes DistilBERTTrainer compatible with classifier.py 

    def predict_proba_text(self, text: str) -> np.ndarray:
        """Single-text probability prediction."""
        return self.predict_proba([text])[0]

    def predict_proba_context(self, text: str, history: list) -> np.ndarray:
        """
        Context-aware prediction with adaptive blending.
        Identical logic to IntentClassifier and DeepIntentTrainer.
        """
        current = self.predict_proba_text(text)
        if not history:
            return current

        hist_vec, weights = np.zeros(3), 0.0
        for i, turn in enumerate(history[-5:]):
            lbl = turn.get("label_id")
            if lbl is not None:
                w = (i + 1) * 0.5
                one_hot = np.zeros(3)
                one_hot[int(lbl)] = 1.0
                hist_vec += w * one_hot
                weights  += w

        if weights == 0:
            return current

        hist_vec /= weights
        current_top_p = float(current.max())
        hist_top_id   = int(np.argmax(hist_vec))
        hist_top_p    = float(hist_vec[hist_top_id])

        # Strong current signal → trust it, barely use context
        if current_top_p >= 0.55:
            return current

        ctx_weight = 0.32 if (hist_top_id == 1 and hist_top_p >= 0.60) else 0.15
        blended = np.clip(
            (1 - ctx_weight) * current + ctx_weight * hist_vec, 1e-9, None
        )
        blended /= blended.sum()

        # Safety: never flatten a confident prediction
        if float(blended.max()) < current_top_p * 0.80:
            return current
        return blended

    @property
    def trained(self):
        return self.model is not None
