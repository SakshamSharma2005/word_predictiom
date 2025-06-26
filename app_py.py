# -*- coding: utf-8 -*-
"""APP.py"""

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model and tokenizer
model = load_model('next_word_predictor.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Prediction function
def predict_next_word(seed_text, model, tokenizer, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=-1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return seed_text + ' ' + word
    return seed_text + ' ...[unknown]'

# Page config
st.set_page_config(page_title="🔮 Shakespearean AI - Next Word Predictor", layout="centered")

# Background + Glass effect
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

# Main content
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">🧠 Next Word Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speak like Shakespeare • Powered by AI • Built using NLP + GRU</div>', unsafe_allow_html=True)

# Input UI
input_text = st.text_input("✍️ Enter a line (from Hamlet or your own):", placeholder="e.g., To be or not to")

# Predict button
if st.button("🔮 Predict the Next Word"):
    if not input_text.strip():
        st.warning("Please enter some text to predict the next word.")
    else:
        max_sequence_len = model.input_shape[1] + 1
        predicted_sentence = predict_next_word(input_text, model, tokenizer, max_sequence_len)
        st.markdown(f'<div class="pred-box">👉 {predicted_sentence}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="border-top: 1px solid #999;">
<div style='text-align: center; color: #ccc; font-size: 0.9rem'>
Made with 💜 by Saksham Sharma • Inspired by Hamlet • #AI #Shakespeare #Streamlit
</div>
""", unsafe_allow_html=True)
