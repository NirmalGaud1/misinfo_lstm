import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_data)
    return tokenizer

# Load model
@st.cache_resource
def load_model_lstm():
    model = load_model("lstm_misinformation_model.h5")
    return model

# Predict function
def predict_misinformation(text, tokenizer, model):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post')
    pred = model.predict(padded)[0][0]
    return 1 if pred >= 0.5 else 0, pred

# Streamlit app
def main():
    st.set_page_config(page_title="Misinformation Detection", layout="centered")
    st.title("🧠 Misinformation Detection using LSTM")

    st.markdown("Enter a tweet or text below to check if it's likely **Misinformation** or **Not**.")

    input_text = st.text_area("Enter Tweet/Text", "", height=150)

    if st.button("Predict"):
        if input_text.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            tokenizer = load_tokenizer()
            model = load_model_lstm()
            label, score = predict_misinformation(input_text, tokenizer, model)

            st.subheader("Prediction")
            if label == 1:
                st.error(f"🛑 This text is likely **Misinformation**. (Confidence: {score:.2f})")
            else:
                st.success(f"✅ This text is likely **Not Misinformation**. (Confidence: {score:.2f})")

    st.markdown("---")
    st.caption("Built with 🧠 LSTM · Streamlit · Keras")

if __name__ == "__main__":
    main()
