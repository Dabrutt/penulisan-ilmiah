import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
import contractions
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Inisialisasi preprocessing ---

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

negation_words = [
    "no", "not", "nor", "neither", "never", "nothing", "nowhere", "hardly",
    "barely", "doesn't", "isn't", "wasn't", "shouldn't", "wouldn't",
    "couldn't", "won't", "can't", "don't", "didn't", "hasn't", "haven't", "hadn't"
]
intensifier_words = [
    "very", "really", "so", "too", "quite", "such", "absolutely", "extremely",
    "totally", "completely", "highly"
]
diminisher_words = [
    "just", "only", "somewhat", "fairly", "slightly", "barely", "hardly"
]
contrast_words = [
    "but", "however", "although", "though", "yet", "even though", "still"
]
important_stopwords = set(
    negation_words + intensifier_words + diminisher_words + contrast_words
)
custom_stopwords = stop_words - important_stopwords

slang_word = {
    "u": "you", "r": "are", "ur": "your", "btw": "by the way",
    "idk": "i do not know", "lol": "laughing out loud", "omg": "oh my god",
    "lmao": "laughing my ass off", "rofl": "rolling on the floor laughing",
    "brb": "be right back", "gtg": "got to go", "imo": "in my opinion",
    "imho": "in my humble opinion", "fyi": "for your information",
    "tbh": "to be honest", "smh": "shaking my head", "np": "no problem",
    "jk": "just kidding", "nvm": "never mind", "bff": "best friend forever",
    "dm": "direct message", "tldr": "too long did not read", "wth": "what the heck",
    "af": "as f*", "ikr": "i know right", "ya": "yeah", "thx": "thanks",
    "ty": "thank you", "plz": "please", "bc": "because", "cuz": "because",
    "tho": "though", "k": "okay", "ok": "okay", "hbu": "how about you",
    "wyd": "what are you doing", "wbu": "what about you", "rn": "right now",
    "bday": "birthday", "gr8": "great", "luv": "love", "xoxo": "hugs and kisses",
    "yall": "you all", "gf": "girl friend"
}

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def cleaningText(text):
    emoji_pattern = re.compile(
        "[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF"
        u"\U00002500-\U00002BEF" u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251" "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', str(text))
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

def casefoldingText(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def expandContractions(text):
    return contractions.fix(text)

def standard_slangwords(text):
    words = text.split()
    return " ".join([slang_word.get(w, w) for w in words])

def tokenizingText(text):
    return word_tokenize(text)

def filteringText(tokens):
    return [word for word in tokens if word not in custom_stopwords]

def lemmatizationText(tokens):
    pos_tags = pos_tag(tokens)
    return [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in pos_tags]

def preprocess_text(text):
    text = cleaningText(text)
    text = casefoldingText(text)
    text = expandContractions(text)
    text = standard_slangwords(text)
    tokens = tokenizingText(text)
    tokens = filteringText(tokens)
    tokens = lemmatizationText(tokens)
    if len(tokens) < 3:
        return None
    return " ".join(tokens)

# --- Load model, tokenizer, label encoder ---

model = tf.keras.models.load_model("lstm_sentiment_model_80_20.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

max_len = 100

# --- Inisialisasi VADER ---

analyzer = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# --- Streamlit UI ---

st.title("Analisis Sentimen dengan LSTM dan VADER")

user_input = st.text_area("Masukkan kalimat:")

if st.button("Analisis Sentimen"):
    if not user_input.strip():
        st.warning("Kalimat tidak boleh kosong.")
    else:
        preprocessed = preprocess_text(user_input)
        if preprocessed is None:
            st.warning("Kalimat terlalu pendek setelah preprocessing. Masukkan kalimat yang lebih panjang.")
        else:
            # Prediksi LSTM
            sequences = tokenizer.texts_to_sequences([preprocessed])
            padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
            prediction = model.predict(padded)
            label_index = np.argmax(prediction)
            sentiment_lstm = label_encoder.inverse_transform([label_index])[0]
            prob_lstm = prediction[0][label_index]

            # Prediksi VADER
            sentiment_vader = vader_sentiment(user_input)

            st.subheader("Hasil Analisis LSTM:")
            st.write(f"Kalimat setelah preprocessing: {preprocessed}")
            st.write(f"Sentimen: **{sentiment_lstm}**")
            st.write(f"Probabilitas: {prob_lstm:.4f}")

            st.subheader("Hasil Analisis VADER:")
            st.write(f"Sentimen: **{sentiment_vader}**")
