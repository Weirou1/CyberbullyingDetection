import joblib
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import ComplementNB
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

_shared_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from postprocessing import (
    has_indirect_bullying_phrase,
    THRESHOLD, adjust_confidence_batch,
    has_personal_target,
)

# ============================================
# 1. Load the preprocessed data
# ============================================
X_train = joblib.load('../shared/X_train.pkl')
X_test = joblib.load('../shared/X_test.pkl')
y_train = joblib.load('../shared/y_train.pkl')
y_test = joblib.load('../shared/y_test.pkl')
tfidf = joblib.load('../shared/tfidf_vectorizer.pkl')

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)


def has_personal_target_text(text):
    tokens = word_tokenize(text.lower())
    return has_personal_target(tokens)


# ============================================
# 2. Load severity_test / severity_train / message_test / intensity_test
# ============================================
severity_test = joblib.load('../shared/severity_test.pkl')
severity_test_arr = severity_test.values if hasattr(severity_test, 'values') else np.asarray(severity_test)
severity_train = joblib.load('../shared/severity_train.pkl')
severity_train_arr = severity_train.values if hasattr(severity_train, 'values') else np.asarray(severity_train)

message_test = joblib.load('../shared/message_test.pkl')
message_test_arr = message_test.values if hasattr(message_test, 'values') else np.asarray(message_test)

intensity_test = joblib.load('../shared/intensity_test.pkl')
intensity_test_arr = intensity_test.values if hasattr(intensity_test, 'values') else np.asarray(intensity_test)

assert len(y_test) == len(severity_test_arr), (
    "Length mismatch between y_test and severity_test — make sure "
    "they were split together (same call, same random_state)."
)
assert len(y_test) == len(message_test_arr), (
    "Length mismatch between y_test and message_test — make sure "
    "preprocessing.py's message_train/message_test split uses the "
    "same random_state=42 / stratify=y as X_train/X_test."
)
assert len(y_test) == len(intensity_test_arr), (
    "Length mismatch between y_test and intensity_test."
)
assert len(y_train) == len(severity_train_arr), (
    "Length mismatch between y_train and severity_train."
)

print("\nComputing has_personal_target on the test set "
      "(uses shared/postprocessing.py's has_personal_target)...")
target_test_arr = np.array([
    has_personal_target_text(str(msg)) for msg in message_test_arr
])
HAS_TARGET_TEST = True
print("Done.")

# ============================================
# 3. Train the model (ComplementNB + balanced sample weights)
# ============================================
sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

nb_model = ComplementNB(alpha=1.0)
nb_model.fit(X_train, y_train, sample_weight=sample_weight)

# ============================================
# 4. Predict raw probabilities on the test set
# ============================================
y_proba_raw = nb_model.predict_proba(X_test)[:, 1]

# ============================================
# 5. Confidence calibration 
# ============================================
X_train2, X_val, y_train2, y_val, severity_train2, severity_val = train_test_split(
    X_train, y_train, severity_train_arr,
    test_size=0.2, random_state=42, stratify=y_train
)

sample_weight2 = compute_sample_weight(class_weight='balanced', y=y_train2)
nb_model_cal = ComplementNB(alpha=1.0)
nb_model_cal.fit(X_train2, y_train2, sample_weight=sample_weight2)
y_proba_val = nb_model_cal.predict_proba(X_val)[:, 1]


def boost_confidence(raw_prob, severity, exponent):
    if severity != "Unclassified / Context-dependent":
        return raw_prob ** exponent
    return raw_prob


def boost_confidence_batch(raw_probs, severities, exponent):
    raw_probs = np.asarray(raw_probs).flatten()
    return np.array([
        boost_confidence(p, s, exponent) for p, s in zip(raw_probs, severities)
    ])


TARGET_RECALL_1 = 0.70  

BOOST_EXPONENT = 1.0  
for exp_candidate in np.arange(1.0, 0.05, -0.02):
    boosted_trial = boost_confidence_batch(y_proba_val, severity_val, exp_candidate)
    preds_trial = (boosted_trial >= THRESHOLD).astype(int)
    recall_1_trial = recall_score(y_val, preds_trial, pos_label=1)
    if recall_1_trial >= TARGET_RECALL_1:
        BOOST_EXPONENT = exp_candidate
        break

print(f"\nDecision threshold (fixed): {THRESHOLD:.3f}")
print(f"Calibrated BOOST_EXPONENT: {BOOST_EXPONENT:.2f} "
      f"(on validation split, target bullying recall: {TARGET_RECALL_1})")
if BOOST_EXPONENT == 1.0:
    print("[NOTE] Validation recall never reached the target even at the "
          "strongest boost tried (0.06) — this means keyword-based boosting "
          "alone cannot close the recall gap. This is expected: ~83% of "
          "messages have no severity keyword match at all (see the severity "
          "level distribution printed by preprocessing.py), so boosting a "
          "score that only applies to the other 17% has a hard ceiling. "
          "The phrase-level override below is what's meant to help with the "
          "no-keyword cases instead.")


# ============================================
# 6. Prediction — before vs after adjustment
# ============================================
y_pred_raw_only = (y_proba_raw > THRESHOLD).astype(int)

print(f"\n--- BEFORE (raw model, threshold={THRESHOLD:.2f}, no boost) ---")
print("Accuracy:", accuracy_score(y_test, y_pred_raw_only))
print(classification_report(y_test, y_pred_raw_only, target_names=['Normal', 'Cyberbullying']))


y_proba_boosted = boost_confidence_batch(y_proba_raw, severity_test_arr, BOOST_EXPONENT)
y_proba_final = adjust_confidence_batch(y_proba_boosted, severity_test_arr, target_test_arr)

y_pred = (y_proba_final > THRESHOLD).astype(int)

phrase_hit_arr = np.array([
    has_indirect_bullying_phrase(str(msg).lower()) for msg in message_test_arr
])
phrase_override_arr = phrase_hit_arr & target_test_arr.astype(bool)
y_pred = np.where(phrase_override_arr, 1, y_pred)

print("\n--- AFTER (severity boost + personal-target discount + "
      f"phrase override, threshold={THRESHOLD:.3f}) ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Normal', 'Cyberbullying']))
print(f"Messages flagged via phrase override alone: {phrase_override_arr.sum()}")


# ============================================
# 7. Evaluation
# ============================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Cyberbullying']))


# ============================================
# 8. Confusion Matrix (using final/adjusted predictions, saved to file)
# ============================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Cyberbullying'],
            yticklabels=['Normal', 'Cyberbullying'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Naive Bayes Confusion Matrix (boost + target-adjustment + phrase override)')
plt.savefig('naive_bayes_confusion_matrix.png', bbox_inches='tight', dpi=150)
plt.show()

print(cm)


# ============================================
# 9. FEATURE： Confidence score distribution histogram 
# ============================================
plt.figure(figsize=(8, 5))
plt.hist(y_proba_final, bins=30, edgecolor='white')
plt.axvline(THRESHOLD, color='red', linestyle='--', label=f'Threshold = {THRESHOLD}')
plt.title('Distribution of Confidence Scores (Test Set) — Naive Bayes')
plt.xlabel('Confidence')
plt.ylabel('Number of Messages')
plt.legend()
plt.savefig('nb_confidence_distribution.png', bbox_inches='tight', dpi=150)
plt.show()
print("\nConfidence distribution saved to 'nb_confidence_distribution.png'")


# ============================================
# 10. Save the trained model
# ============================================
joblib.dump(nb_model, 'naive_bayes_model.pkl')
joblib.dump(THRESHOLD, 'decision_threshold.pkl')
joblib.dump(BOOST_EXPONENT, 'nb_boost_exponent.pkl')
print(f"Calibrated BOOST_EXPONENT saved as 'nb_boost_exponent.pkl' ({BOOST_EXPONENT:.2f})")
print("\nModel saved as 'naive_bayes_model.pkl'")
print(f"Decision threshold saved as 'decision_threshold.pkl' ({THRESHOLD:.3f})")


# ============================================
# 11. Accuracy by severity level (and by target)
# ============================================
comparison = pd.DataFrame({
    'true_label': y_test.values if hasattr(y_test, 'values') else np.asarray(y_test),
    'predicted_label': y_pred,
    'confidence_score': y_proba_final,
    'severity_level': severity_test_arr,
    'had_personal_target': target_test_arr,
    'phrase_override': phrase_override_arr,
    'intensity_score': intensity_test_arr,
})

comparison['correct'] = comparison['true_label'] == comparison['predicted_label']

print("\nNaive Bayes Accuracy by severity level:")
print(comparison.groupby('severity_level')['correct'].agg(accuracy='mean', count='count'))

print("\nAccuracy by whether message had a personal target:")
print(comparison.groupby('had_personal_target')['correct'].agg(accuracy='mean', count='count'))

print("\nAccuracy by intensity score bucket:")
intensity_bins = [0, 0.001, 0.1, 0.3, 1.0]
intensity_labels = ['0 (no trigger words)', '0-0.1 (low)', '0.1-0.3 (medium)', '0.3-1.0 (high)']
comparison['intensity_bucket'] = pd.cut(
    comparison['intensity_score'], bins=intensity_bins, labels=intensity_labels, include_lowest=True
)
print(comparison.groupby('intensity_bucket', observed=True)['correct'].agg(accuracy='mean', count='count'))

print("\nSeverity level counts in test set:")
print(comparison['severity_level'].value_counts())

comparison.to_csv('nb_severity_analysis.csv', index=False)
print("\nSeverity breakdown saved to 'nb_severity_analysis.csv'")
