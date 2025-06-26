# -*- coding: utf-8 -*-
"""app.py"""

import streamlit as st
import numpy as np
import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model('next_word_predictor.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Helper: Clean input (keep letters and spaces)
def clean_input(text):
    cleaned = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned.lower().strip()

# Multi-word generation logic
def generate_next_words(seed_text, model, tokenizer, max_sequence_len, num_words=5):
    cleaned_text = clean_input(seed_text)
    if not cleaned_text:
        return None, "⚠️ Please enter valid alphabetic text."

        for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([cleaned_text])[0]
        if not token_list:
            return cleaned_text, "⚠️ Not enough context to predict. Try more meaningful text."

        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)[0]

        next_word = None
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                next_word = word
                break

        if next_word:
            cleaned_text += ' ' + next_word
        else:
            cleaned_text += ' ...[unknown]'
            break

    return f"📝 *Shakespeare-style continuation:* \n\n👉 **{cleaned_text}**", None


# Page Config
st.set_page_config(page_title="🔮 Shakespearean AI - Next Word Predictor", layout="centered")

# Stylish CSS
st.markdown("""
    <style>
    body {
        background-image: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        background-attachment: fixed;
    }
    .main {
        background: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 15px;
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
        color: white;
    }
    .title {
        font-size: 2.5rem;
        text-align: center;
        color: #ffffff;
        animation: glow 2s ease-in-out infinite alternate;
    }
    .subtitle {
        font-size: 1.1rem;
        text-align: center;
        color: #cccccc;
        margin-bottom: 30px;
    }
    @keyframes glow {
        from {
            text-shadow: 0 0 10px #fff, 0 0 20px #9d4edd, 0 0 30px #7b2cbf;
        }
        to {
            text-shadow: 0 0 20px #fff, 0 0 30px #7b2cbf, 0 0 40px #5a189a;
        }
    }
    .pred-box {
        background: linear-gradient(135deg, #7b2cbf, #3c096c);
        padding: 1rem;
        border-radius: 10px;
        color: #ffffff;
        font-weight: 500;
        font-size: 1.2rem;
        text-align: center;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main container
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">🧠 Next Word Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speak like Shakespeare • Powered by AI • Built using NLP + LSTM</div>', unsafe_allow_html=True)

# Input field
input_text = st.text_input("✍️ Enter a line (from Hamlet or your own):", placeholder="e.g., To be or not to")

# Word count selector
num_words = st.slider("🔢 Number of words to generate", min_value=1, max_value=20, value=5)

# Predict button
if st.button("🔮 Predict the Next Words"):
    max_sequence_len = model.input_shape[1] + 1
    result, error = generate_next_words(input_text, model, tokenizer, max_sequence_len, num_words)

    if error:
        st.warning(error)
    else:
        cleaned = clean_input(input_text)
        if cleaned != input_text.strip().lower():
            st.markdown(f"<small style='color: #bbb;'>Cleaned input used: <code>{cleaned}</code></small>", unsafe_allow_html=True)
        st.markdown(f'<div class="pred-box">👉 {result}</div>', unsafe_allow_html=True)

# Close UI
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="border-top: 1px solid #999;">
<div style='text-align: center; color: #ccc; font-size: 0.9rem'>
Made with 💜 by Saksham Sharma • Inspired by Hamlet • #AI #Shakespeare #Streamlit
</div>
""", unsafe_allow_html=True)
