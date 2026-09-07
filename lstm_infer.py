import os
import re
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib

_here = os.path.dirname(os.path.abspath(__file__))

_shared_dir = os.path.abspath(os.path.join(_here, '..', 'shared'))
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)

from postprocessing import (
    clean_text, replace_emoji_with_tokens,
    negation_words, classify_severity_live,
    calculate_intensity_score, get_trigger_words, has_indirect_bullying_phrase,
    THRESHOLD,classify_with_review_flag, adjust_confidence, has_personal_target,
)



def _path(filename):
    return os.path.join(_here, filename)


_missing = [f for f in ['lstm_model.keras', 'lstm_tokenizer.pkl']
            if not os.path.exists(_path(f))]
if _missing:
    raise FileNotFoundError(
        f"Missing {_missing} in {_here}. Run lstm_model.py (with "
        "FORCE_RETRAIN=True at least once) before starting the UI."
    )

_model = load_model(_path('lstm_model.keras'))
_tokenizer = joblib.load(_path('lstm_tokenizer.pkl'))

MAX_LEN = 80

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words('english')) - negation_words


def predict(text, threshold=THRESHOLD):
    text_lower = replace_emoji_with_tokens(text).lower()

    raw_tokens = word_tokenize(text_lower)
    target_present = has_personal_target(raw_tokens)

    cleaned = clean_text(text_lower)
    tokens = word_tokenize(cleaned)

    tokens_clean = [_lemmatizer.lemmatize(w) for w in tokens if w not in _stop_words]
    final = ' '.join(tokens_clean)

    severity = classify_severity_live(final, target_present)
    intensity = calculate_intensity_score(final)
    triggers = get_trigger_words(final)

    seq = _tokenizer.texts_to_sequences([final])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

    raw_prediction = float(_model.predict(padded, verbose=0)[0][0])
    prediction = adjust_confidence(raw_prediction, severity, target_present, threshold)
    has_harmful_word = severity != "Unclassified / Context-dependent"

    phrase_hit = has_indirect_bullying_phrase(text.lower())
    phrase_override = phrase_hit and target_present
    if phrase_override:
        severity = "Level 3 - Severe (Indirect Bullying / Social Exclusion)"

    label = classify_with_review_flag(prediction, has_harmful_word, threshold)
    if phrase_override:
        label = "Cyberbullying"

    return {
        "model": "LSTM",
        "text": text,
        "label": label,
        "display_label": label,
        "confidence": prediction,
        "raw_confidence": raw_prediction,
        "threshold": threshold,
        "severity": severity,
        "intensity_score": intensity,
        "has_personal_target": target_present,
        "phrase_override": phrase_override,
        "trigger_words": triggers,
    }