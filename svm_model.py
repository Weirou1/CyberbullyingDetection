import os
import sys
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

_shared_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)

from postprocessing import (
    clean_text, negation_words, classify_severity, has_indirect_bullying_phrase,
    THRESHOLD, adjust_confidence_batch,
    has_personal_target,
)

# ============================================
# Load the preprocessed data (produced by preprocessing.py).
# ============================================
X_train = joblib.load('../shared/X_train.pkl')
X_test = joblib.load('../shared/X_test.pkl')
y_train = joblib.load('../shared/y_train.pkl')
y_test = joblib.load('../shared/y_test.pkl')
tfidf = joblib.load('../shared/tfidf_vectorizer.pkl')
severity_test = joblib.load('../shared/severity_test.pkl')
intensity_test = joblib.load('../shared/intensity_test.pkl')

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
print("Severity test size:", severity_test.shape)
print("Intensity test size:", intensity_test.shape)

# ============================================
# Shared text-cleaning helpers 
# ============================================
stop_words = set(stopwords.words('english')) - negation_words
lemmatizer = WordNetLemmatizer()

# ============================================
# Reconstruct target_test 
# ============================================
df_check = pd.read_csv('../shared/AI_dataset.csv')
df_check = df_check.dropna(subset=['Message', 'Label'])
df_check = df_check.reset_index(drop=True)

df_check['cleaned_message'] = df_check['Message'].apply(clean_text)
df_check['cleaned_message'] = df_check['cleaned_message'].apply(lambda x: x.lower())
df_check['tokens'] = df_check['cleaned_message'].apply(word_tokenize)
df_check['tokens_no_stopwords'] = df_check['tokens'].apply(
    lambda tokens: [w for w in tokens if w not in stop_words]
)
df_check['lemmatized_tokens'] = df_check['tokens_no_stopwords'].apply(
    lambda tokens: [lemmatizer.lemmatize(w) for w in tokens]
)
df_check['final_text'] = df_check['lemmatized_tokens'].apply(lambda tokens: ' '.join(tokens))
df_check = df_check[df_check['final_text'].str.strip() != '']

df_check['raw_tokens'] = df_check['Message'].astype(str).str.lower().apply(word_tokenize)
df_check['has_personal_target'] = df_check['raw_tokens'].apply(has_personal_target)
df_check['severity_level'] = df_check['final_text'].apply(classify_severity)  # for the severity pie chart (Feature 4)

y_check = df_check['Label']
_, target_test = train_test_split(
    df_check['has_personal_target'],
    test_size=0.3, random_state=42, stratify=y_check
)
_, message_test = train_test_split(
    df_check['Message'],
    test_size=0.3, random_state=42, stratify=y_check
)

assert len(target_test) == X_test.shape[0], \
    f"MISMATCH: target_test has {len(target_test)} rows but X_test has {X_test.shape[0]} rows! " \
    f"preprocessing.py must have changed - re-check dataset/cleaning steps before trusting results."
assert len(message_test) == X_test.shape[0], \
    f"MISMATCH: message_test has {len(message_test)} rows but X_test has {X_test.shape[0]} rows!"
print("target_test reconstructed successfully, aligned with X_test:", target_test.shape)


# ============================================
# Hyperparameter tuning (GridSearchCV).
# ============================================
param_grid = {'C': [0.01, 0.1, 1, 10]}

grid_search = GridSearchCV(
    LinearSVC(class_weight='balanced'),
    param_grid,
    scoring='f1',
    cv=5
)
grid_search.fit(X_train, y_train)

print("\n===== Hyperparameter Tuning (GridSearchCV) =====")
print("Best C:", grid_search.best_params_)
print("Best CV F1 score:", grid_search.best_score_)

best_C = grid_search.best_params_['C']

# ============================================
# Build and train the SVM model, calibrated for probability output.
# ============================================
base_svm = LinearSVC(C=best_C, class_weight='balanced')
svm_model = CalibratedClassifierCV(base_svm, method='sigmoid', cv=5)
svm_model.fit(X_train, y_train)
print("Model training completed (calibrated for probability output)!")

# ============================================
# Cross-validation (on training set).
# ============================================
cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='f1')
print("\n===== 5-Fold Cross-Validation (on training set) =====")
print("F1 scores per fold:", cv_scores)
print("Mean F1:", cv_scores.mean())
print("Std F1 :", cv_scores.std())

# ============================================
# Evaluate the model.
# ============================================
y_pred_prob_raw = svm_model.predict_proba(X_test)[:, 1]

severity_test_arr = severity_test.values if hasattr(severity_test, 'values') else severity_test
target_test_arr = target_test.values if hasattr(target_test, 'values') else target_test
intensity_test_arr = intensity_test.values if hasattr(intensity_test, 'values') else intensity_test
message_test_arr = message_test.values if hasattr(message_test, 'values') else message_test

y_pred_prob_adjusted = adjust_confidence_batch(y_pred_prob_raw, severity_test_arr, target_test_arr)
y_pred = (y_pred_prob_adjusted > THRESHOLD).astype(int)

phrase_hit_arr = np.array([
    has_indirect_bullying_phrase(str(msg).lower()) for msg in message_test_arr
])
phrase_override_arr = phrase_hit_arr & target_test_arr.astype(bool)
y_pred = np.where(phrase_override_arr, 1, y_pred)

print("\n--- BEFORE adjustment (raw model output) ---")
y_pred_raw_only = (y_pred_prob_raw > THRESHOLD).astype(int)
print("Accuracy:", accuracy_score(y_test, y_pred_raw_only))
print(classification_report(y_test, y_pred_raw_only, target_names=['Normal', 'Cyberbullying']))

print("\n--- AFTER adjustment (with personal-target rule + phrase override) ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Normal', 'Cyberbullying']))
print(f"Messages flagged via phrase override alone: {phrase_override_arr.sum()}")

print("\n-------- Threshold sweep on ADJUSTED predictions --------")
for t in [0.4, 0.5, 0.6, 0.7, 0.8]:
    preds_t = (y_pred_prob_adjusted > t).astype(int)
    p = precision_score(y_test, preds_t, zero_division=0)
    r = recall_score(y_test, preds_t, zero_division=0)
    a = accuracy_score(y_test, preds_t)
    print(f"threshold={t:.1f} -> precision={p:.3f} recall={r:.3f} accuracy={a:.3f}")
print("----------------------------------------------------------\n")

# ============================================
# Confusion matrix 
# ============================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Cyberbullying'],
            yticklabels=['Normal', 'Cyberbullying'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix (target-adjustment + phrase override)')
plt.savefig('svm_confusion_matrix.png', bbox_inches='tight', dpi=150)
plt.show()

print(cm)

# ============================================
# Feature: Confidence score distribution histogram 
# ============================================
plt.figure(figsize=(8, 5))
plt.hist(y_pred_prob_adjusted, bins=30, edgecolor='white')
plt.axvline(THRESHOLD, color='red', linestyle='--', label=f'Threshold = {THRESHOLD}')
plt.title('Distribution of Confidence Scores (Test Set) — SVM')
plt.xlabel('Confidence')
plt.ylabel('Number of Messages')
plt.legend()
plt.savefig('svm_confidence_distribution.png', bbox_inches='tight', dpi=150)
plt.show()
print("Confidence distribution saved to svm_confidence_distribution.png")

# ============================================
# Save the trained model.
# ============================================
joblib.dump(svm_model, 'svm_model.pkl')
print("Model saved as svm_model.pkl")

# ============================================
# Model accuracy by severity level AND by personal-target presence 
# ============================================
comparison = pd.DataFrame({
    'true_label': y_test.values if hasattr(y_test, 'values') else y_test,
    'predicted_label': y_pred,
    'confidence_score': y_pred_prob_adjusted,
    'severity_level': severity_test_arr,
    'had_personal_target': target_test_arr,
    'phrase_override': phrase_override_arr,
    'intensity_score': intensity_test_arr
})
comparison['correct'] = comparison['true_label'] == comparison['predicted_label']

print("\nSVM Accuracy by severity level:")
print(comparison.groupby('severity_level')['correct'].mean())

print("\nAccuracy by whether message had a personal target:")
print(comparison.groupby('had_personal_target')['correct'].mean())

print("\nAccuracy by intensity score bucket:")
intensity_bins = [0, 0.001, 0.1, 0.3, 1.0]
intensity_labels = ['0 (no trigger words)', '0-0.1 (low)', '0.1-0.3 (medium)', '0.3-1.0 (high)']
comparison['intensity_bucket'] = pd.cut(
    comparison['intensity_score'], bins=intensity_bins, labels=intensity_labels, include_lowest=True
)
print(comparison.groupby('intensity_bucket', observed=True)['correct'].agg(accuracy='mean', count='count'))

print("\nSeverity level counts in test set:")
print(comparison['severity_level'].value_counts())

comparison.to_csv('svm_severity_analysis.csv', index=False)
print("\nSeverity analysis saved to svm_severity_analysis.csv")
