# -*- coding: utf-8 -*-
"""APP.py"""

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the LSTM Model
model = load_model('next_word_predictor.h5')

# Load the tokenizer
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

    seed_text += ' ' + output_word
    return seed_text

# Streamlit app
st.title("Next Word Prediction")
input_text = st.text_input("Enter the text")

if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(input_text, model, tokenizer, max_sequence_len)
    st.write(f'Next word: {next_word}')
