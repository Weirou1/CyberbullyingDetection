import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import random
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# Load the dataset.
# ============================================
df = pd.read_csv('AI_dataset.csv')
print("Shape before dropping missing values:", df.shape)

# ============================================
# Handle missing values 
# ============================================
df = df.dropna(subset=['Message', 'Label'])
df = df.reset_index(drop=True)
print("Shape after dropping missing values:", df.shape)

# ============================================
# Emoji handling 
# ============================================
EMOJI_TOKEN_MAP = {
    # hostile / mocking -- these get added to severity_keywords (Level 2)
    "🤡": "emojiclown", "💀": "emojiskull", "🤮": "emojivomit",
    "🖕": "emojimiddlefinger", "😡": "emojiangry", "🤬": "emojicursing",
    "👎": "emojithumbsdown", "🙄": "emojieyeroll",
    "🔪": "emojiknife",
    # friendly / neutral -- ordinary vocabulary, not treated as harmful
    "😊": "emojismile", "🙂": "emojislightsmile", "👍": "emojithumbsup",
    "❤️": "emojiheart", "🎉": "emojiparty",
}
HOSTILE_EMOJI_TOKENS = {"emojiclown", "emojiskull", "emojivomit", "emojimiddlefinger",
                         "emojiangry", "emojicursing", "emojithumbsdown", "emojieyeroll",
                         "emojiknife"}
FRIENDLY_EMOJI = ["😊", "🙂", "👍", "❤️", "🎉"]
HOSTILE_EMOJI = ["🤡", "💀", "🤮", "🖕", "😡", "🤬", "👎", "🙄", "🔪"]

def replace_emoji_with_tokens(text):
    for emoji_char, token in EMOJI_TOKEN_MAP.items():
        text = text.replace(emoji_char, f" {token} ")
    return text

# ============================================
# Remove punctuation, special characters, and numbers.
# ============================================
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_message'] = df['Message'].apply(replace_emoji_with_tokens)
df['cleaned_message'] = df['cleaned_message'].apply(lambda x: x.lower())
df['cleaned_message'] = df['cleaned_message'].apply(clean_text)
df['tokens'] = df['cleaned_message'].apply(word_tokenize)

# ============================================
# Preserve negation words during stopword removal
# ============================================
negation_words = {"not", "dont", "didnt", "doesnt", "wont", "never",
                   "no", "cant", "isnt", "wasnt", "wouldnt", "shouldnt", "aint"}
stop_words = set(stopwords.words('english')) - negation_words
df['tokens_no_stopwords'] = df['tokens'].apply(
    lambda tokens: [word for word in tokens if word not in stop_words]
)

lemmatizer = WordNetLemmatizer()
df['lemmatized_tokens'] = df['tokens_no_stopwords'].apply(
    lambda tokens: [lemmatizer.lemmatize(word) for word in tokens]
)

df['final_text'] = df['lemmatized_tokens'].apply(lambda tokens: ' '.join(tokens))

df = df[df['final_text'].str.strip() != '']
print("Shape after removing empty rows:", df.shape)

# ============================================
# Severity Level Classification.
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


def classify_severity(text):
    words = text.split()
    for level, keywords in severity_keywords.items():
        for i, w in enumerate(words):
            if w in keywords:
                # Check the 3 words before this keyword for a negation word
                window = words[max(0, i-3):i]
                if any(neg in window for neg in negation_words):
                    continue  # skip this match, it appears to be negated
                return level
    return "Unclassified / Context-dependent"

df['severity_level'] = df['final_text'].apply(classify_severity)
print("Severity level distribution:")
print(df['severity_level'].value_counts())

# ============================================
# Intensity Score A continuous 0-1 score
# ============================================
_all_trigger_words = set()
for _kws in severity_keywords.values():
    _all_trigger_words.update(_kws)

def calculate_intensity_score(final_text):
    words = final_text.split()
    if len(words) == 0:
        return 0.0
    trigger_count = sum(1 for w in words if w in _all_trigger_words)
    return round(trigger_count / len(words), 4)

df['intensity_score'] = df['final_text'].apply(calculate_intensity_score)
print("Intensity score summary:")
print(df['intensity_score'].describe())

# ============================================
# Export severity + intensity classification as CSV.
# ============================================
df[['Message', 'Label', 'severity_level', 'intensity_score', 'final_text']].to_csv(
    'severity_classified_messages.csv', index=False
)
print("Severity + intensity classification CSV exported!")

try:
    from wordcloud import WordCloud

    cyberbullying_text = ' '.join(df[df['Label'] == 1]['final_text'].astype(str))
    normal_text = ' '.join(df[df['Label'] == 0]['final_text'].astype(str))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(
        WordCloud(width=800, height=400, background_color='white',
                  colormap='Reds').generate(cyberbullying_text),
        interpolation='bilinear'
    )
    axes[0].axis('off')
    axes[0].set_title('Most Common Words: Cyberbullying Messages')

    axes[1].imshow(
        WordCloud(width=800, height=400, background_color='white',
                  colormap='Blues').generate(normal_text),
        interpolation='bilinear'
    )
    axes[1].axis('off')
    axes[1].set_title('Most Common Words: Normal Messages')

    plt.tight_layout()
    plt.savefig('wordcloud_comparison.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("Word cloud saved to wordcloud_comparison.png")
except ImportError:
    print("[NOTE] 'wordcloud' package not installed -- skipping word "
          "cloud. Run: pip install wordcloud")

# Severity pie chart
plt.figure(figsize=(9, 7))
severity_counts = df['severity_level'].value_counts()
wedges, _texts, _autotexts = plt.pie(
    severity_counts,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette('pastel'),
    pctdistance=0.8,
)
plt.legend(
    wedges, severity_counts.index,
    title="Severity Level",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=9,
)
plt.title('Severity Level Distribution (Full Dataset)')
plt.tight_layout()
plt.savefig('severity_distribution.png', bbox_inches='tight', dpi=150)
plt.show()
print("Severity distribution pie chart saved to severity_distribution.png")

# ============================================
# TF-IDF Feature Extraction.
# ============================================
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['final_text'])
y = df['Label']

# ============================================
# Split into training and testing sets (70% train, 30% test).
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ============================================
# Also split severity_level, intensity_score
# ============================================
_, severity_test = train_test_split(
    df['severity_level'], test_size=0.3, random_state=42, stratify=y
)
severity_train, _ = train_test_split(
    df['severity_level'], test_size=0.3, random_state=42, stratify=y
)
_, intensity_test = train_test_split(
    df['intensity_score'], test_size=0.3, random_state=42, stratify=y
)
message_train, message_test = train_test_split(
    df['Message'], test_size=0.3, random_state=42, stratify=y
)

# ============================================
# Save everything so the whole group can use the same data.
# ============================================
joblib.dump(X_train, 'X_train.pkl')
joblib.dump(X_test, 'X_test.pkl')
joblib.dump(y_train, 'y_train.pkl')
joblib.dump(y_test, 'y_test.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(severity_test, 'severity_test.pkl')
joblib.dump(severity_train, 'severity_train.pkl')
joblib.dump(intensity_test, 'intensity_test.pkl')
joblib.dump(message_train, 'message_train.pkl')
joblib.dump(message_test, 'message_test.pkl')

print("Files saved successfully!")

# ============================================
# Emoji demo set 
# ============================================
random.seed(42)
demo_sample = df.sample(n=300, random_state=42).copy().reset_index(drop=True)
demo_sample['has_emoji'] = False
demo_sample['emoji_added'] = ""

n_emoji = int(len(demo_sample) * 0.09) 
emoji_indices = random.sample(range(len(demo_sample)), n_emoji)

for idx in emoji_indices:
    if demo_sample.loc[idx, 'Label'] == 1:
        emoji_char = random.choice(HOSTILE_EMOJI)
    else:
        emoji_char = random.choice(FRIENDLY_EMOJI)
    demo_sample.loc[idx, 'Message'] = str(demo_sample.loc[idx, 'Message']) + " " + emoji_char
    demo_sample.loc[idx, 'has_emoji'] = True
    demo_sample.loc[idx, 'emoji_added'] = emoji_char

demo_sample[['Message', 'Label', 'has_emoji', 'emoji_added']].to_csv(
    'emoji_demo_messages.csv', index=False
)
print(f"\nEmoji demo set saved: emoji_demo_messages.csv "
      f"({n_emoji}/{len(demo_sample)} messages have emoji, ~9%). "
      "Demo/robustness testing only -- NOT used for training or the "
      "official accuracy/precision/recall comparison.")