import pandas as pd
import os
import sys
import random
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


_shared_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)

from postprocessing import (
    clean_text, replace_emoji_with_tokens, negation_words, calculate_intensity_score,
    has_indirect_bullying_phrase, THRESHOLD, REVIEW_MARGIN,
    has_personal_target, classify_severity, adjust_confidence_batch,
)

# ============================================
# Fix all random seeds so training is reproducible across runs.
# ============================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================
# Set this to True only when you've changed the dataset and need a full reprocess + retrain.
# ============================================
FORCE_RETRAIN = False

if FORCE_RETRAIN:
    print("FORCE_RETRAIN=True -> removing all cached files...")
    for f in ['lstm_model.keras', 'df_preprocessed_lstm.pkl', 'lstm_X_train.pkl',
              'lstm_X_test.pkl', 'lstm_y_train.pkl', 'lstm_y_test.pkl',
              'lstm_tokenizer.pkl', 'lstm_severity_test.pkl', 'lstm_target_test.pkl',
              'lstm_intensity_test.pkl', 'lstm_X_test_text.pkl', 'lstm_message_test.pkl']:
        if os.path.exists(f):
            os.remove(f)
            print(f"Deleted {f}")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

max_words = 15000
max_len = 80

stop_words = stop_words - negation_words

# ============================================
# 1. Load preprocessed data (skip cleaning if already saved).
# ============================================
if os.path.exists('df_preprocessed_lstm.pkl'):
    print("Loading previously preprocessed data...")
    df = joblib.load('df_preprocessed_lstm.pkl')
else:
    print("No saved preprocessed data found. Running preprocessing from scratch...")
    df = pd.read_csv('../shared/AI_dataset.csv')

    df['cleaned_message'] = df['Message'].apply(replace_emoji_with_tokens)
    df['cleaned_message'] = df['cleaned_message'].apply(lambda x: x.lower())
    df['cleaned_message'] = df['cleaned_message'].apply(clean_text)
    df['tokens'] = df['cleaned_message'].apply(word_tokenize)

    df['tokens_no_stopwords'] = df['tokens'].apply(
        lambda tokens: [word for word in tokens if word not in stop_words]
    )

    df['lemmatized_tokens'] = df['tokens_no_stopwords'].apply(
        lambda tokens: [lemmatizer.lemmatize(word) for word in tokens]
    )

    df['final_text'] = df['lemmatized_tokens'].apply(lambda tokens: ' '.join(tokens))
    df = df[df['final_text'].str.strip() != '']
    print("Shape after removing empty rows:", df.shape)

    joblib.dump(df, 'df_preprocessed_lstm.pkl')
    print("Preprocessed dataframe saved!")

# ============================================
# 2. Feature preparation
# ============================================
if 'severity_level' not in df.columns:
    df['severity_level'] = df['final_text'].apply(classify_severity)
    joblib.dump(df, 'df_preprocessed_lstm.pkl')

if 'intensity_score' not in df.columns:
    df['intensity_score'] = df['final_text'].apply(calculate_intensity_score)
    joblib.dump(df, 'df_preprocessed_lstm.pkl')
    print("Intensity scores calculated.")

if 'has_personal_target' not in df.columns:
    df['raw_tokens_for_target'] = df['Message'].astype(str).str.lower().apply(word_tokenize)
    df['has_personal_target'] = df['raw_tokens_for_target'].apply(has_personal_target)
    joblib.dump(df, 'df_preprocessed_lstm.pkl')

# ============================================
# 3. Tokenize + split into train/test (skip if already saved)
# ============================================
if os.path.exists('lstm_X_train.pkl'):
    print("Loading previously tokenized sequences...")
    X_train_pad = joblib.load('lstm_X_train.pkl')
    X_test_pad = joblib.load('lstm_X_test.pkl')
    y_train = joblib.load('lstm_y_train.pkl')
    y_test = joblib.load('lstm_y_test.pkl')
    tokenizer = joblib.load('lstm_tokenizer.pkl')
    severity_test_lstm = joblib.load('lstm_severity_test.pkl')
    target_test_lstm = joblib.load('lstm_target_test.pkl')
    intensity_test_lstm = joblib.load('lstm_intensity_test.pkl')
    X_test_text = joblib.load('lstm_X_test_text.pkl')
    message_test_lstm = joblib.load('lstm_message_test.pkl')
else:
    print("No saved sequences found. Running tokenization from scratch...")

    (X_train_text, X_test_text,
     y_train, y_test,
     severity_train_lstm, severity_test_lstm,
     target_train_lstm, target_test_lstm,
     intensity_train_lstm, intensity_test_lstm,
     message_train_lstm, message_test_lstm) = train_test_split(
        df['final_text'], df['Label'], df['severity_level'], df['has_personal_target'],
        df['intensity_score'], df['Message'],
        test_size=0.3, random_state=42, stratify=df['Label']
    )

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)

    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

    print("Train shape after padding:", X_train_pad.shape)
    print("Test shape after padding:", X_test_pad.shape)

    joblib.dump(X_train_pad, 'lstm_X_train.pkl')
    joblib.dump(X_test_pad, 'lstm_X_test.pkl')
    joblib.dump(y_train, 'lstm_y_train.pkl')
    joblib.dump(y_test, 'lstm_y_test.pkl')
    joblib.dump(tokenizer, 'lstm_tokenizer.pkl')
    joblib.dump(severity_test_lstm, 'lstm_severity_test.pkl')
    joblib.dump(target_test_lstm, 'lstm_target_test.pkl')
    joblib.dump(intensity_test_lstm, 'lstm_intensity_test.pkl')
    joblib.dump(X_test_text, 'lstm_X_test_text.pkl')
    joblib.dump(message_test_lstm, 'lstm_message_test.pkl')
    print("Tokenized sequences saved!")

assert len(y_test) == len(severity_test_lstm) == len(target_test_lstm) == len(intensity_test_lstm) == len(message_test_lstm), (
    "Length mismatch between y_test, severity_test_lstm, target_test_lstm, intensity_test_lstm, message_test_lstm. "
    "Set FORCE_RETRAIN=True and re-run."
)

# ============================================
# 4. Train the model (load if cached, otherwise build + train + save)
# ============================================
history = None

if os.path.exists('lstm_model.keras'):
    print("Loading previously trained model...")
    model = load_model('lstm_model.keras')
else:
    print("No saved model found. Building and training from scratch...")

    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
    model.add(Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)))
    model.add(Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3)))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weight_dict)

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train_pad, y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.1,
        class_weight=class_weight_dict,
        callbacks=[early_stop]
    )

    model.save('lstm_model.keras')
    print("Model saved successfully!")

# ============================================
# 5. Evaluate the model — before vs after adjustment, threshold sweep
# ============================================
y_pred_prob_raw = model.predict(X_test_pad)

severity_test_arr = severity_test_lstm.values if hasattr(severity_test_lstm, 'values') else severity_test_lstm
target_test_arr = target_test_lstm.values if hasattr(target_test_lstm, 'values') else target_test_lstm
message_test_arr = message_test_lstm.values if hasattr(message_test_lstm, 'values') else message_test_lstm

y_pred_prob_adjusted = adjust_confidence_batch(y_pred_prob_raw, severity_test_arr, target_test_arr)
y_pred = (y_pred_prob_adjusted > THRESHOLD).astype(int)

phrase_hit_arr = np.array([
    has_indirect_bullying_phrase(str(msg).lower()) for msg in message_test_arr
])
phrase_override_arr = phrase_hit_arr & target_test_arr.astype(bool)
y_pred = np.where(phrase_override_arr, 1, y_pred)

print("\n--- BEFORE adjustment (raw model output) ---")
y_pred_raw_only = (y_pred_prob_raw.flatten() > THRESHOLD).astype(int)
print("Accuracy:", accuracy_score(y_test, y_pred_raw_only))
print(classification_report(y_test, y_pred_raw_only, target_names=['Normal', 'Cyberbullying']))

print("\n--- AFTER adjustment (with personal-target rule + phrase override) ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Normal', 'Cyberbullying']))
print(f"Messages flagged via phrase override alone: {phrase_override_arr.sum()}")

# Threshold sweep on the adjusted predictions
print("\n--- Threshold sweep on ADJUSTED predictions ---")
for t in [0.4, 0.5, 0.6, 0.7, 0.8]:
    preds_t = (y_pred_prob_adjusted > t).astype(int)
    p = precision_score(y_test, preds_t, zero_division=0)
    r = recall_score(y_test, preds_t, zero_division=0)
    a = accuracy_score(y_test, preds_t)
    print(f"threshold={t:.1f} -> precision={p:.3f} recall={r:.3f} accuracy={a:.3f}")
print("------------------------------------------------\n")

# ============================================
# 6. Find real test-set messages that land in the "Needs Human Review" band
# ============================================
review_mask = (
    (y_pred_prob_adjusted > (THRESHOLD - REVIEW_MARGIN)) &
    (y_pred_prob_adjusted < (THRESHOLD + REVIEW_MARGIN))
)
print(f"Messages in the 'Needs Human Review' band: {review_mask.sum()} / {len(y_pred_prob_adjusted)}")
if review_mask.sum() > 0:
    X_test_text_arr = X_test_text.values if hasattr(X_test_text, 'values') else np.asarray(X_test_text)
    print("Example messages you can test via lstm_infer.py's predict() (or the Streamlit UI) to demo this feature:")
    for ex in X_test_text_arr[review_mask][:5]:
        print(f"  - {ex!r}")
else:
    print("No test-set messages landed in this band with the current trained "
          "model -- confidence scores are too polarized. Try REVIEW_MARGIN=0.20 "
          "temporarily to demo the feature, or note this as a finding: the "
          "model rarely produces genuinely uncertain predictions.")

# ============================================
# 7. Confusion Matrix.
# ============================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Cyberbullying'],
            yticklabels=['Normal', 'Cyberbullying'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('LSTM Confusion Matrix (target-adjustment + phrase override)')
plt.savefig('lstm_confusion_matrix.png', bbox_inches='tight', dpi=150)
plt.show()

print(cm)

# ============================================
# 8. Confidence Score Distribution Histogram.
# ============================================
plt.figure(figsize=(8, 5))
plt.hist(y_pred_prob_adjusted, bins=30, edgecolor='white')
plt.axvline(THRESHOLD, color='red', linestyle='--', label=f'Threshold = {THRESHOLD}')
plt.title('Distribution of Confidence Scores (Test Set) — LSTM')
plt.xlabel('Confidence')
plt.ylabel('Number of Messages')
plt.legend()
plt.savefig('lstm_confidence_distribution.png', bbox_inches='tight', dpi=150)
plt.show()
print("Confidence distribution saved to lstm_confidence_distribution.png")

# ============================================
# 9. Training History 
# ============================================
if history is not None:
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('lstm_training_history.png', bbox_inches='tight', dpi=150)
    plt.show()
else:
    print("Model loaded from cache — skipping training history plot (see saved lstm_training_history.png).")

# ============================================
# 10. Accuracy by severity level, personal-target presence, and
# intensity score bucket
# ============================================
comparison = pd.DataFrame({
    'true_label': y_test.values,
    'predicted_label': y_pred.flatten(),
    'confidence_score': y_pred_prob_adjusted.flatten(),
    'severity_level': severity_test_lstm.values,
    'had_personal_target': target_test_arr,
    'phrase_override': phrase_override_arr,
    'intensity_score': intensity_test_lstm.values
})

comparison['correct'] = comparison['true_label'] == comparison['predicted_label']

print("\nLSTM Accuracy by severity level:")
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

comparison.to_csv('lstm_severity_analysis.csv', index=False)
print("\nSeverity analysis saved to lstm_severity_analysis.csv")