import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.layers import LSTM as OriginalLSTM

class CustomLSTM(OriginalLSTM):
    def __init__(self, *args, **kwargs):
        # Remove 'time_major' from kwargs if present to avoid errors during initialization
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_json_str = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json_str)
    return tokenizer

# Load model with custom LSTM to handle 'time_major' parameter
@st.cache_resource
def load_model_lstm():
    custom_objects = {'LSTM': CustomLSTM}
    try:
        model = load_model("lstm_misinformation_model.h5", custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_misinformation(text, tokenizer, model):
    if model is None:
        return -1, 0.0  # Model failed to load
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post')
    pred = model.predict(padded)[0][0]
    return 1 if pred >= 0.5 else 0, pred

def main():
    st.set_page_config(page_title="Misinformation Detection", layout="centered")
    st.title("🧠 Misinformation Detection using LSTM")
    st.markdown("Enter a tweet or text below to check if it's likely **Misinformation** or **Not**.")

    input_text = st.text_area("Enter Tweet/Text", "", height=150)

    if st.button("Predict"):
        if not input_text.strip():
            st.warning("Please enter some text to classify.")
        else:
            tokenizer = load_tokenizer()
            model = load_model_lstm()
            if model:
                label, score = predict_misinformation(input_text, tokenizer, model)
                st.subheader("Prediction")
                if label == 1:
                    st.error(f"🛑 Likely **Misinformation** (Confidence: {score:.2f})")
                else:
                    st.success(f"✅ Likely **Not Misinformation** (Confidence: {score:.2f})")

    st.markdown("---")
    st.caption("Built with 🧠 LSTM · Streamlit · Keras")

if __name__ == "__main__":
    main()
