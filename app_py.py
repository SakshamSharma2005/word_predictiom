# -*- coding: utf-8 -*-
"""app.py"""

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model and tokenizer
model = load_model('next_word_predictor.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Multi-word generation function
def generate_next_words(seed_text, model, tokenizer, max_sequence_len, num_words=5):
    seed_text = seed_text.strip()
    if not seed_text:
        return "⚠️ Please enter valid text to predict."

    original_seed = seed_text

    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        if not token_list:
            return "⚠️ Not enough context to predict. Try a longer or different input."

        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)[0]

        next_word = None
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                next_word = word
                break

        if next_word:
            seed_text += ' ' + next_word
        else:
            seed_text += ' ...[unknown]'
            break

    return f"📝 *Shakespeare-style continuation:* \n\n👉 **{seed_text}**"

# Streamlit Page Config
st.set_page_config(page_title="🔮 Shakespearean AI - Next Word Predictor", layout="centered")

# Enhanced UI Style
st.markdown("""
    <style>
    body {
        background-image: linear-gradient(to right, #1f1c2c, #928dab);
        background-attachment: fixed;
    }
    .main {
        background: rgba(255, 255, 255, 0.07);
        padding: 2rem;
        border-radius: 20px;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .title {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        color: #ffffff;
        animation: pulse 2s infinite;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #dddddd;
        margin-bottom: 30px;
        font-style: italic;
    }
    .pred-box {
        background: linear-gradient(145deg, #5e60ce, #3a0ca3);
        padding: 1.5rem;
        border-radius: 15px;
        color: #ffffff;
        font-weight: 500;
        font-size: 1.3rem;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 0 10px #5e60ce;
    }
    @keyframes pulse {
        0% {
            text-shadow: 0 0 10px #fff, 0 0 20px #7b2cbf;
        }
        100% {
            text-shadow: 0 0 20px #fff, 0 0 30px #9d4edd;
        }
    }
    </style>
""", unsafe_allow_html=True)

# App Wrapper
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">🧠 Next Word Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speak like Shakespeare • Powered by AI • Built using NLP + LSTM</div>', unsafe_allow_html=True)

# Input Field
input_text = st.text_input("✍️ Enter a line (from Hamlet or your own):", placeholder="e.g., To be or not to")

# Word count slider
num_words = st.slider("🔢 Number of words to generate", min_value=1, max_value=20, value=5)

# Predict Button
if st.button("🔮 Predict the Next Words"):
    max_sequence_len = model.input_shape[1] + 1
    prediction = generate_next_words(input_text, model, tokenizer, max_sequence_len, num_words=num_words)
    st.markdown(f'<div class="pred-box">{prediction}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="border-top: 1px solid #999;">
<div style='text-align: center; color: #ccc; font-size: 0.9rem'>
Made with 💜 by Saksham Sharma • Inspired by Hamlet • #AI #Shakespeare #Streamlit
</div>
""", unsafe_allow_html=True)
