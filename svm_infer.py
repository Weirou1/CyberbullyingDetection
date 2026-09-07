import os
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
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


_missing = [f for f in ['svm_model.pkl'] if not os.path.exists(_path(f))]
_missing += [f for f in ['tfidf_vectorizer.pkl']
             if not os.path.exists(_path(os.path.join('..', 'shared', f)))]
if _missing:
    raise FileNotFoundError(
        f"Missing {_missing}. Run svm_model.py fully at least once "
        "(it also needs ../shared/preprocessing.py to have been run) "
        "before starting the UI."
    )

_svm_model = joblib.load(_path('svm_model.pkl'))
_tfidf = joblib.load(_path(os.path.join('..', 'shared', 'tfidf_vectorizer.pkl')))

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words('english')) - negation_words

def predict(text, threshold=THRESHOLD):
    text_lower = replace_emoji_with_tokens(text).lower()

    raw_tokens = word_tokenize(text_lower)
    target_present = has_personal_target(raw_tokens)

    cleaned_msg = clean_text(text_lower)
    tokens = word_tokenize(cleaned_msg)
    tokens_clean = [_lemmatizer.lemmatize(w) for w in tokens if w not in _stop_words]
    final = ' '.join(tokens_clean)

    severity = classify_severity_live(final, target_present)
    intensity = calculate_intensity_score(final)
    trigger_words = get_trigger_words(final)

    vector = _tfidf.transform([final])
    raw_prediction = float(_svm_model.predict_proba(vector)[0][1])
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
        "model": "SVM",
        "text": text,
        "label": label,
        "display_label": label,
        "confidence": float(prediction),
        "raw_confidence": raw_prediction,
        "threshold": threshold,
        "severity": severity,
        "intensity_score": intensity,
        "has_personal_target": target_present,
        "phrase_override": phrase_override,
        "trigger_words": trigger_words,
    }