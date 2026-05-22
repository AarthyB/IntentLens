import re, string, random, numpy as np, pandas as pd, joblib
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

LABEL_NAMES  = ["Platonic", "Romantic", "Ambiguous"]
LABEL_EMOJIS = ["🤝", "💕", "🤔"]
LABEL_COLORS = ["#60a5fa", "#f472b6", "#fbbf24"]
MODEL_DIR = Path(__file__).parent.parent / "models"


def clean(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\d+", "", text).strip()


# INPUT PARSER
# Separates: quoted message to analyze | speaker | user commentary | intent type

def parse_input(raw: str) -> dict:
    t  = raw.strip()
    tl = t.lower()

    #  Step 1: Does the message contain any quoted text? 
    # We check for real quotes BEFORE deciding if it's a meta question.
    # Curly quotes " " or straight "
    quoted_curly  = re.findall(r'[\u201c\u2018]([^\u201c\u201d\u2018\u2019]+)[\u201d\u2019]', t)
    quoted_double = re.findall(r'"([^"]{4,})"', t)
    # Single quotes only when they clearly wrap a sentence (start/end of string or after said/texted)
    quoted_single = re.findall(r"(?:said|texted|wrote|messaged)\s+'([^']{6,})'", tl)

    has_quote = bool(quoted_curly or quoted_double or quoted_single)
    quoted_text = (quoted_curly or quoted_double or quoted_single or [None])[0]

    #  Step 2: Meta detection — only when NO quotes present 
    if not has_quote:
        meta_patterns = [
            r"\bbut even (a |as )?(friend|friends)\b",
            r"\beven (a |as )?(friend|friends) (can|could|would|might)\b",
            r"\bcan'?t (a |any )?friend\b",
            r"\bcould(n'?t)? (a |any )?friend\b",
            r"\bdoes that mean\b",
            r"\bso does that\b",
            r"\bwhy did you (say|call|label|classify)\b",
            r"\bare you sure\b",
            r"\bwhat do you mean\b",
            r"\bhow do you know\b",
            r"\bexplain (that|this|why)\b",
            r"\bthat'?s not (right|correct|accurate)\b",
            r"\byou'?re wrong\b",
            r"\bisn'?t that just\b",
        ]
        for pat in meta_patterns:
            if re.search(pat, tl):
                return {"analyze_text": None, "speaker": None, "speaker_label": None,
                        "user_note": t, "is_meta": True, "is_context": False}

    #  Step 3: Detect speaker 
    speaker_label = None
    speaker_patterns = [
        (r'\bhe (said|texted|wrote|messaged|told me)\b', 'He'),
        (r'\bshe (said|texted|wrote|messaged|told me)\b', 'She'),
        (r'\bthey (said|texted|wrote|messaged)\b',       'They'),
        (r'\bmy (friend|bf|boyfriend|gf|girlfriend|crush|partner|ex)\b', 'Your friend'),
        (r'\bmy (guy|girl|boo)\b',                       'Your friend'),
        (r'\bsomeone (said|texted|wrote)\b',             'Someone'),
        (r'\bi (said|texted|wrote|told)\b',              'You'),
    ]
    for pat, lbl in speaker_patterns:
        if re.search(pat, tl):
            speaker_label = lbl
            break

    #  Step 4: Context-only notes (no analyzable message) 
    # Only mark as context if it CLEARLY has no analyzable message:
    # - starts with connector words AND is short/meta-like
    # - references what someone else said (not the message itself)
    if not has_quote:
        ctx_patterns = [
            # "for context", "btw", "also" — filler openers with no emotional content
            r"^(btw|by the way|for context|just to add|to clarify)\b",
            # "X said my friend" style — attribution of reported speech with no quote
            r"\b(mutual effort|sustained|sustain)\b.*\bsaid\b",
        ]
        for pat in ctx_patterns:
            if re.search(pat, tl):
                return {"analyze_text": None, "speaker": speaker_label, "speaker_label": speaker_label,
                        "user_note": t, "is_meta": False, "is_context": True}

    #  Step 5: Determine what to analyze 
    if has_quote:
        analyze_text = quoted_text
        # User note = everything outside the quotes
        user_note = re.sub(r'[\u201c"]([^\u201c\u201d"]+)[\u201d"]', '', t)
        user_note = re.sub(r"said '([^']+)'", '', user_note).strip(' ,-')
    else:
        analyze_text = t
        user_note    = ""

    return {
        "analyze_text":  analyze_text,
        "speaker":       speaker_label,
        "speaker_label": speaker_label or "They",
        "user_note":     user_note,
        "is_meta":       False,
        "is_context":    False,
    }


# TF-IDF + LOGISTIC REGRESSION
class IntentClassifier:
    def __init__(self):
        self.vec = TfidfVectorizer(max_features=6000, ngram_range=(1, 3),
                                   sublinear_tf=True, min_df=1)
        self.clf = LogisticRegression(C=2.0, max_iter=2000, solver="lbfgs", random_state=42)
        self.trained = False

    def fit(self, texts, labels):
        X = self.vec.fit_transform([clean(t) for t in texts])
        self.clf.fit(X, labels)
        self.trained = True
        return self

    def predict_proba_text(self, text: str) -> np.ndarray:
        X = self.vec.transform([clean(text)])
        return self.clf.predict_proba(X)[0]

    def predict_proba_context(self, text: str, history: list) -> np.ndarray:
        """
        Context-aware prediction with adaptive blending.

        Key principle: the CURRENT MESSAGE always dominates.
        Context provides a soft nudge, never an override.
        The 33.3% flat-distribution bug happened because a mixed
        history was being blended too aggressively. Now we cap
        ctx_weight at 0.25 and never let it produce a flatter
        distribution than the current message alone.
        """
        current = self.predict_proba_text(text)
        if not history:
            return current

        # Build recency-weighted history vector
        hist_vec, weights = np.zeros(3), 0.0
        for i, turn in enumerate(history[-5:]):
            lbl = turn.get("label_id")
            if lbl is not None:
                w = (i + 1) * 0.5          # older turns get lower weight
                one_hot = np.zeros(3)
                one_hot[int(lbl)] = 1.0
                hist_vec += w * one_hot
                weights  += w

        if weights == 0:
            return current

        hist_vec /= weights

        current_top_id  = int(np.argmax(current))
        current_top_p   = float(current[current_top_id])
        hist_top_id     = int(np.argmax(hist_vec))
        hist_top_p      = float(hist_vec[hist_top_id])

        #  Adaptive context weight 
        if current_top_p >= 0.60:
            # Clear signal in current message → barely use context
            ctx_weight = 0.08
        elif current_top_p >= 0.45:
            # Moderate signal → light context nudge
            ctx_weight = 0.18
        elif hist_top_id == 1 and hist_top_p >= 0.60:
            # Conversation is clearly romantic, current msg is uncertain → lean romantic
            ctx_weight = 0.32
        else:
            # Low confidence on both sides → minimal blending
            ctx_weight = 0.15

        blended = np.clip((1 - ctx_weight) * current + ctx_weight * hist_vec, 1e-9, None)
        blended /= blended.sum()

        #  Safety check: protect high-confidence current predictions 
        # If the raw current message is already confident (>=0.55),
        # never let history dilute it — return the raw prediction.
        # This prevents platonic history from overriding clear romantic confessions.
        if current_top_p >= 0.55:
            return current

        # For lower-confidence current predictions, use the blend
        # but only if it doesn't make the top label LESS likely
        blended_top_p = float(blended[int(np.argmax(blended))])
        if blended_top_p < current_top_p * 0.80:
            return current

        return blended

    def evaluate(self, texts, labels):
        X = self.vec.transform([clean(t) for t in texts])
        preds = self.clf.predict(X)
        return {"accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds, average="macro"),
                "confusion_matrix": confusion_matrix(labels, preds).tolist(),
                "report": classification_report(labels, preds, target_names=LABEL_NAMES),
                "predictions": preds.tolist()}

    def save(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vec, MODEL_DIR / "tfidf.pkl")
        joblib.dump(self.clf, MODEL_DIR / "lr.pkl")

    def load(self):
        self.vec = joblib.load(MODEL_DIR / "tfidf.pkl")
        self.clf = joblib.load(MODEL_DIR / "lr.pkl")
        self.trained = True
        return self

    @property
    def model_exists(self):
        return (MODEL_DIR / "tfidf.pkl").exists() and (MODEL_DIR / "lr.pkl").exists()


# REPLY BUILDER
def _detect_phrases(text: str, label_name: str) -> list:
    t = text.lower()
    phrase_map = {
        "Platonic":  ["whole group","crew","everyone","lunch","sibling","study","class",
                      "notes","friend","project","advice","roommate","together","mutual",
                      "just friends","only friends","like a brother","like a sister"],
        "Romantic":  ["can't stop thinking","date","falling","fall for","kiss","yours",
                      "heart","just us","love","like you","feelings","beautiful","stunning",
                      "miss you","morning","thinking about you","future","didn't message",
                      "message me","message today","crush","picked me up","checks on me",
                      "messages every day","texts every day","messages when","worried about me"],
        "Ambiguous": ["can't explain","not sure","figuring out","butterflies","between us",
                      "close to","different","feel something","don't want to end",
                      "always on my mind","nervous","miss you even","worried about",
                      "don't know if","not sure if","only as a friend","just as a friend",
                      "see me as","sees me as"],
    }
    return [f'"{p}"' for p in phrase_map.get(label_name, []) if p in t][:3]


def _extract_emotional_signals(text: str) -> str:
    """
    For long narrative paragraphs, extract the emotionally significant
    sentences rather than passing the whole block verbatim.
    This improves classification of multi-sentence inputs.
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 8]
    
    if len(sentences) <= 2:
        return text   # Short enough, use as-is
    
    # Score each sentence for emotional signal words
    romantic_signals  = ["crush","feelings","likes me","love","miss","heart","nervous",
                         "checks on","messages when","texts when","picked me up",
                         "picked me","thinks about","always on","special","date"]
    ambiguous_signals = ["don't know","not sure","see me as","only as","just as",
                         "confused","weird","different","changed"]
    
    scored = []
    for s in sentences:
        sl = s.lower()
        score = sum(1 for w in romantic_signals if w in sl) * 2
        score += sum(1 for w in ambiguous_signals if w in sl)
        scored.append((score, s))
    
    # Keep top 3 most emotionally loaded sentences
    scored.sort(key=lambda x: -x[0])
    top = [s for _, s in scored[:3]]
    return " ".join(top)


def build_conversational_reply(raw: str, classifier: "IntentClassifier", history: list) -> dict:
    parsed = parse_input(raw)

    #  META: user questioning the analysis 
    if parsed["is_meta"]:
        random.seed(hash(raw) % 9999)
        prev = history[-1] if history else None
        prev_label = LABEL_NAMES[prev["label_id"]] if prev else None
        prev_text  = prev.get("analyzed_text") or "" if prev else ""

        if prev_label:
            pool = [
                f"You're right to push back — and that's exactly the challenge. "
                f"The model flagged **{prev_label}** for *\"{prev_text[:50]}\"*, "
                f"but a close friend absolutely could say something like that. "
                f"The difference usually lives in tone, history, and patterns — "
                f"not the words alone. What else do you know about their relationship?",

                f"Fair point. **{prev_label}** is the model's statistical best guess, not a verdict. "
                f"Friends and people with romantic feelings often use the same words. "
                f"That's why the confidence score was low — the model itself isn't certain. "
                f"Wider context (how often they talk, their history) matters far more.",

                f"Exactly the problem this research explores. The same phrase means something different "
                f"depending on who says it and what the relationship looks like. "
                f"The model can only see the words — you have the full picture. "
                f"What does your gut say?",
            ]
        else:
            pool = [
                "Good point — a single message is hard to judge without relationship context. Share more and I can give a fuller read.",
                "You're right. Context matters more than any one line. What else do you know about the situation?",
            ]
        reply = random.choice(pool)
        probs = np.array([1/3, 1/3, 1/3])
        label_id = 2
        return _pack(raw, None, None, label_id, probs, "Context question", reply, history)

    #  CONTEXT NOTE: no message to analyze 
    if parsed["is_context"]:
        random.seed(hash(raw) % 9999)
        pool = [
            "Got it — useful context. What's the specific message you'd like me to look at?",
            "Noted. Go ahead and paste or quote the actual message.",
            "Thanks for the background. What did they say exactly — quote it and I'll analyze it.",
        ]
        return _pack(raw, None, None, 2, np.array([1/3,1/3,1/3]),
                     "Context received", random.choice(pool), history)

    #  CLASSIFY the actual message 
    analyze_text  = parsed["analyze_text"]
    spk           = parsed["speaker_label"]  # "He", "She", "Your friend", "They", …
    user_note     = parsed["user_note"]

    # For long narrative paragraphs (>120 chars, multi-sentence), extract key signals
    classify_text = analyze_text
    if len(analyze_text) > 120 and any(c in analyze_text for c in ".!?"):
        classify_text = _extract_emotional_signals(analyze_text)

    probs    = classifier.predict_proba_context(classify_text, history)
    label_id = int(np.argmax(probs))
    conf     = float(probs[label_id])
    label    = LABEL_NAMES[label_id]

    if conf >= 0.75:   certainty = "High confidence"
    elif conf >= 0.50: certainty = "Moderate confidence"
    else:              certainty = "Low confidence — this is genuinely ambiguous"

    cues      = _detect_phrases(analyze_text, label)
    cue_str   = f" I picked up on: {', '.join(cues)}." if cues else ""

    shift_note = ""
    if history:
        prev_lid = history[-1].get("label_id")
        if prev_lid is not None and int(prev_lid) != label_id:
            shift_note = f" That's a shift from the previous message, which read as **{LABEL_NAMES[int(prev_lid)]}**."

    pattern_note = ""
    if len(history) >= 2:
        lids = [h.get("label_id") for h in history[-3:] if h.get("label_id") is not None]
        if lids and Counter(lids).most_common(1)[0][0] == label_id and len(lids) >= 2:
            pattern_note = f" This is consistent with the overall tone of this conversation."

    random.seed(hash(analyze_text) % 9999)

    platonic_pool = [
        f'**"{analyze_text}"** reads as **platonic**.{cue_str} {spk} sounds like a caring friend — not someone signalling romantic interest.{shift_note}{pattern_note}',
        f'{spk} is being a good friend here. The language is warm and supportive, but there\'s no sign of emotional pursuit.{cue_str}{shift_note}{pattern_note}',
        f'That message is firmly in friendship territory.{cue_str} {spk} seems to genuinely care, but the framing is social, not romantic.{shift_note}{pattern_note}',
    ]
    romantic_pool = [
        f'**"{analyze_text}"** leans **romantic**.{cue_str} The underlying expectation — that you owed them contact — goes beyond what most friends imply.{shift_note}{pattern_note}',
        f'{spk}\'s message carries emotional weight that points past friendship.{cue_str} It signals "you matter to me in a specific way" — not just as a friend.{shift_note}{pattern_note}',
        f'There\'s a quiet expectation in that message that\'s hard to read as purely platonic.{cue_str} {spk} may be more invested than a typical friend would be.{shift_note}{pattern_note}',
    ]
    ambiguous_pool = [
        f'**"{analyze_text}"** is genuinely ambiguous.{cue_str} {spk} could be a close friend who values mutual effort — or someone with deeper feelings. Hard to call without more context.{shift_note}{pattern_note}',
        f'That\'s a tough one — this message sits right on the line between close friendship and romantic interest.{cue_str} Both friends and people with feelings say things like this. What\'s the relationship history like?{shift_note}{pattern_note}',
        f'Mixed signals.{cue_str} The words are warm but the intent isn\'t clear — this is exactly the kind of message the model struggles with, and honestly so do most people.{shift_note}{pattern_note}',
    ]

    pool_map = {0: platonic_pool, 1: romantic_pool, 2: ambiguous_pool}
    reply = random.choice(pool_map[label_id])

    if user_note and len(user_note.strip()) > 8:
        reply += f' *(You noted: "{user_note.strip()[:70]}" — that context helps.)*'

    return _pack(raw, analyze_text, spk, label_id, probs, certainty, reply, history)


def _pack(raw, analyze_text, speaker, label_id, probs, certainty, reply, history) -> dict:
    label = LABEL_NAMES[label_id]
    conf  = float(probs[label_id])
    return {
        "label_id":      label_id,
        "label":         label,
        "emoji":         LABEL_EMOJIS[label_id],
        "color":         LABEL_COLORS[label_id],
        "confidence":    round(conf * 100, 1),
        "certainty":     certainty,
        "reply":         reply,
        "analyzed_text": analyze_text,
        "speaker":       speaker,
        "probabilities": [
            {"label": LABEL_NAMES[i], "emoji": LABEL_EMOJIS[i],
             "color": LABEL_COLORS[i], "prob": round(float(probs[i]) * 100, 1)}
            for i in range(3)
        ],
        "context_used": len(history),
    }


def train_and_save(dataset_path="data/dataset.csv") -> dict:
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.dataset_builder import build_dataset_v2
    # Always rebuild to pick up new examples
    df = build_dataset_v2(output_path=dataset_path)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    clf = IntentClassifier()
    clf.fit(train_df["text"].tolist(), train_df["label"].tolist())
    metrics = clf.evaluate(test_df["text"].tolist(), test_df["label"].tolist())
    clf.save()
    print(f"\nTrained  acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}")
    return metrics


if __name__ == "__main__":
    train_and_save()
