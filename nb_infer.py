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


_missing = [f for f in ['naive_bayes_model.pkl', 'nb_boost_exponent.pkl']
            if not os.path.exists(_path(f))]
_missing += [f for f in ['tfidf_vectorizer.pkl']
             if not os.path.exists(_path(os.path.join('..', 'shared', f)))]
if _missing:
    raise FileNotFoundError(
        f"Missing {_missing}. Run naive_bayes_model.py fully at least "
        "once (it also needs ../shared/preprocessing.py to have been "
        "run) before starting the UI."
    )

_nb_model = joblib.load(_path('naive_bayes_model.pkl'))
_tfidf = joblib.load(_path(os.path.join('..', 'shared', 'tfidf_vectorizer.pkl')))
BOOST_EXPONENT = joblib.load(_path('nb_boost_exponent.pkl'))

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words('english')) - negation_words

def boost_confidence(raw_prob, severity, exponent):
    if severity != "Unclassified / Context-dependent":
        return raw_prob ** exponent
    return raw_prob


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
    trigger_words = get_trigger_words(final)

    vector = _tfidf.transform([final])
    raw_prediction = float(_nb_model.predict_proba(vector)[0][1])

    boosted = boost_confidence(raw_prediction, severity, BOOST_EXPONENT)
    prediction = adjust_confidence(boosted, severity, target_present, threshold)

    has_harmful_word = severity != "Unclassified / Context-dependent"

    phrase_hit = has_indirect_bullying_phrase(text.lower())
    phrase_override = phrase_hit and target_present

    if phrase_override:
        severity = "Level 3 - Severe (Indirect Bullying / Social Exclusion)"

    label = "Cyberbullying" if (phrase_override or (prediction >= threshold and has_harmful_word)) else "Normal"

    review_flag = classify_with_review_flag(prediction, has_harmful_word, threshold=threshold)

    return {
        "model": "Naive Bayes",
        "text": text,
        "label": label,
        "review_flag": review_flag,
        "display_label": review_flag if review_flag == "Needs Human Review" else label,
        "confidence": float(prediction),
        "raw_confidence": raw_prediction,
        "threshold": float(threshold),
        "severity": severity,
        "intensity_score": intensity,
        "has_personal_target": target_present,
        "phrase_override": phrase_override,
        "trigger_words": trigger_words,
    }