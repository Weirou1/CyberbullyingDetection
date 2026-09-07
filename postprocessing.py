import re
import numpy as np


# ============================================
# Text cleaning
# ============================================
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================
# Emoji handling
# ============================================
EMOJI_TOKEN_MAP = {
    "🤡": "emojiclown", "💀": "emojiskull", "🤮": "emojivomit",
    "🖕": "emojimiddlefinger", "😡": "emojiangry", "🤬": "emojicursing",
    "👎": "emojithumbsdown", "🙄": "emojieyeroll", "🔪": "emojiknife",
    "😊": "emojismile", "🙂": "emojislightsmile", "👍": "emojithumbsup",
    "❤️": "emojiheart", "🎉": "emojiparty",
}


def replace_emoji_with_tokens(text):
    for emoji_char, token in EMOJI_TOKEN_MAP.items():
        text = text.replace(emoji_char, f" {token} ")
    return text


# ============================================
# Negation words
# ============================================
negation_words = {
    "not", "dont", "didnt", "doesnt", "wont", "never",
    "no", "cant", "isnt", "wasnt", "wouldnt", "shouldnt", "aint"
}


# ============================================
# Severity level classification
# ============================================
severity_keywords = {
    "Level 3 - Severe (Threats & Hate Speech)": [
        "kill", "die", "hurt", "destroy", "rape", "hitler",
        "nigger", "nigga", "faggot", "fag", "jew"
    ],
    "Level 2 - Moderate (Insults & Degradation)": [
        "stupid", "moron", "idiot", "dumb", "pig",
        "asshole", "wanker", "fat", "cunt", "bitch", "dickhead",
        "loser", "bastard", "fucker", "cocksucker", "cock",
        "bully", "bullying", "bullied", "harass", "harassing", "harassed",
        "emojiclown", "emojiskull", "emojivomit", "emojimiddlefinger",
        "emojiangry", "emojicursing", "emojithumbsdown", "emojieyeroll"
    ],
    "Level 1 - Mild (General Profanity)": [
        "shit", "bullshit", "fuck", "fucking", "damn", "suck", "sucks",
        "hell", "ass", "motherfucker", "motherfucking", "fuk"
    ]
}

_all_trigger_words = set()
for _kws in severity_keywords.values():
    _all_trigger_words.update(_kws)

UNCONDITIONAL_HATE_WORDS = {"hitler", "nigger", "nigga", "faggot", "fag", "jew"}


def classify_severity(text):
    words = text.split()
    for level, keywords in severity_keywords.items():
        for i, w in enumerate(words):
            if w in keywords:
                window = words[max(0, i - 3):i]
                if any(neg in window for neg in negation_words):
                    continue
                return level
    return "Unclassified / Context-dependent"


def classify_severity_live(text, target_present=False):
    words = text.split()
    for level, keywords in severity_keywords.items():
        for i, w in enumerate(words):
            if w in keywords:
                window = words[max(0, i - 3):i]
                if any(neg in window for neg in negation_words):
                    continue
                if w not in UNCONDITIONAL_HATE_WORDS and not target_present:
                    continue
                return level
    return "Unclassified / Context-dependent"


def calculate_intensity_score(final_text):
    words = final_text.split()
    if len(words) == 0:
        return 0.0
    trigger_count = sum(1 for w in words if w in _all_trigger_words)
    return round(trigger_count / len(words), 4)


def get_trigger_words(final_text):
    """Explainability helper: which words triggered a severity
    classification, and at which level."""
    words = final_text.split()
    found = []
    for level, keywords in severity_keywords.items():
        for w in words:
            if w in keywords and (w, level) not in found:
                found.append((w, level))
    return found


# ============================================
# Indirect / phrase-level bullying detection
# ============================================
INDIRECT_BULLYING_PATTERNS = [
    r"\bnobody\s+likes?\s+(?:you|u|ya)\b",
    r"\bno\s*one\s+likes?\s+(?:you|u|ya)\b",
    r"\beverybody\s+hates?\s+(?:you|u|ya)\b",
    r"\beveryone\s+hates?\s+(?:you|u|ya)\b",
    r"\bnobody\s+wants?\s+(?:you|u|ya)\b",
    r"\bno\s*one\s+wants?\s+(?:you|u|ya)\b",
    r"\bnobody\s+cares?\s+about\s+(?:you|u|ya)\b",
    r"\bno\s*one\s+cares?\s+about\s+(?:you|u|ya)\b",
    r"\bjust\s+disappear\b",
    r"\bgo\s+disappear\b",
    r"\byou\s+should\s+disappear\b",
    r"\bkill\s+(?:yourself|urself|ur\s*self)\b",
    r"\bkys\b",
    r"\bgo\s+die\b",
    r"\byou\s+should\s+die\b",
    r"\bno\s*one\s+would\s+miss\s+(?:you|u|ya)\b",
    r"\bnobody\s+would\s+miss\s+(?:you|u|ya)\b",
    r"\bworld\s+would\s+be\s+better\s+without\s+(?:you|u|ya)\b",
    r"\bwhy\s+don'?t\s+(?:you|u|ya)\s+just\s+leave\b",
]


def has_indirect_bullying_phrase(raw_text_lower):
    return any(re.search(pattern, raw_text_lower) for pattern in INDIRECT_BULLYING_PATTERNS)


# ============================================
# Decision thresholds 
# ============================================
THRESHOLD = 0.70
REVIEW_MARGIN = 0.10
CONFIDENCE_DISCOUNT = 0.8
BOUNDARY_MARGIN = 0.15

def classify_with_review_flag(prediction, has_harmful_word, threshold=THRESHOLD, margin=REVIEW_MARGIN):
    if not has_harmful_word:
        return "Normal"
    if abs(prediction - threshold) <= margin:
        return "Needs Human Review"
    return "Cyberbullying" if prediction > threshold else "Normal"


def adjust_confidence(raw_prediction, severity, target_present, threshold=None):
    if threshold is None:
        threshold = THRESHOLD
    if severity in ("Level 1 - Mild (General Profanity)",
                     "Level 2 - Moderate (Insults & Degradation)"):
        if not target_present and abs(raw_prediction - threshold) < BOUNDARY_MARGIN:
            return raw_prediction * CONFIDENCE_DISCOUNT
    return raw_prediction


def adjust_confidence_batch(raw_predictions, severities, target_flags, threshold=None):
    raw_predictions = np.asarray(raw_predictions).flatten()
    adjusted = raw_predictions.copy()
    for i in range(len(adjusted)):
        adjusted[i] = adjust_confidence(raw_predictions[i], severities[i], target_flags[i], threshold)
    return adjusted


# ============================================
# Personal-target detection
# ============================================
try:
    from nltk import pos_tag
except ImportError:
    pos_tag = None

import nltk as _nltk
for _tagger_name in ('averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger'):
    try:
        _nltk.data.find(f'taggers/{_tagger_name}')
    except LookupError:
        try:
            _nltk.download(_tagger_name)
        except Exception:
            pass

personal_target_pronouns = {
    "you", "your", "youre", "ur", "u", "yall",
    "he", "she", "him", "her", "they", "them", "their",
    "yourself"
}
target_nouns = {
    "girl", "boy", "guy", "dude", "kid",
    "woman", "man", "person", "girlfriend", "boyfriend"
}
all_harmful_words = {
    word for keywords in severity_keywords.values() for word in keywords
}
PROXIMITY_WINDOW = 3
COPULA_VERBS = {"is", "was", "are", "were", "be", "been", "being", "am"}
SUBJECT_TAGS = {"NNP", "NNPS"}
CLAUSE_BOUNDARIES = {",", ".", ";", "and", "but", "or"}
HARM_VERBS_OBJECT_TARGETED = {"kill", "hurt", "destroy", "rape"}


def has_personal_target(tokens, proximity_window=PROXIMITY_WINDOW):
    if any(re.match(r'@\w+', w) for w in tokens):
        return True
    for i, w in enumerate(tokens[:-1]):
        if w == '@' and re.match(r'\w+', tokens[i + 1]):
            return True

    words_only_early = [w for w in tokens if w not in CLAUSE_BOUNDARIES]

    non_directional_harmful_present = any(
        w in all_harmful_words and w not in HARM_VERBS_OBJECT_TARGETED
        for w in words_only_early
    )
    harm_verb_present = any(w in HARM_VERBS_OBJECT_TARGETED for w in words_only_early)

    if any(w in personal_target_pronouns for w in words_only_early):
        if not harm_verb_present or non_directional_harmful_present:
            return True

    tags = None
    if pos_tag is not None:
        try:
            tags = pos_tag(tokens)
        except LookupError:
            tags = None

    if tags:
        for i, (word, _tag) in enumerate(tags):
            if word in all_harmful_words:
                for j in range(i - 1, -1, -1):
                    prev_word, _prev_tag = tags[j]
                    if prev_word in COPULA_VERBS:
                        for k in range(j - 1, -1, -1):
                            subj_word, subj_tag = tags[k]
                            if subj_word in CLAUSE_BOUNDARIES:
                                break
                            if (subj_tag in SUBJECT_TAGS
                                    or subj_word in target_nouns
                                    or subj_word in personal_target_pronouns):
                                return True
                        break
                    if prev_word in CLAUSE_BOUNDARIES:
                        break

    target_words = target_nouns | personal_target_pronouns
    words_only = words_only_early
    for i, word in enumerate(words_only):
        if word in all_harmful_words:
            if word in HARM_VERBS_OBJECT_TARGETED:
                nearby = words_only[i + 1:i + proximity_window + 1]
            else:
                nearby = (
                    words_only[max(0, i - proximity_window):i]
                    + words_only[i + 1:i + proximity_window + 1]
                )
            if any(t in target_words for t in nearby):
                return True

    return False