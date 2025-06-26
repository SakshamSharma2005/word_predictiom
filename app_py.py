# -*- coding: utf-8 -*-
"""APP.py"""

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the model and tokenizer
model = load_model('next_word_predictor.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(seed_text, model, tokenizer, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=-1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            output_word = word
            break
    else:
        output_word = "<unknown>"

    return seed_text + ' ' + output_word

# ------------------ Streamlit UI ------------------

# Set custom page config
st.set_page_config(page_title="Shakespeare Next Word Predictor", page_icon="🧠", layout="centered")

# Apply some custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 2rem;
        border-radius: 10px;
    }
    .title {
        text-align: center;
        font-size: 2.2rem;
        color: #4B0082;
    }
    .subtext {
        text-align: center;
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<div class="title">🧠 Next Word Prediction Model</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Inspired by Shakespeare\'s <i>Hamlet</i> • Built using NLP + GRU</div>', unsafe_allow_html=True)

# Input section
st.markdown("### Enter a line to predict the next word:")
input_text = st.text_input("Your text", placeholder="e.g., To be or not to...")

# Prediction button
if st.button("🔮 Predict Next Word"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        max_sequence_len = model.input_shape[1] + 1
        result = predict_next_word(input_text, model, tokenizer, max_sequence_len)
        st.success(f"📘 **Predicted Output:** `{result}`")

# Footer
st.markdown("---")
st.markdown(
    "<small>Made with ❤️ using Streamlit • Model by Saksham Sharma</small>",
    unsafe_allow_html=True
)
