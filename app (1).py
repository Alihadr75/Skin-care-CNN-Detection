import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Skin Lesion Image Classifier", page_icon="🔬", layout="centered")

MODEL_PATH = Path("model.h5")
CLASS_NAMES_PATH = Path("class_names.json")

# These labels are the common HAM10000 class abbreviations.
# Prefer class_names.json created during training, because class order must
# exactly match the model's output order.
DEFAULT_CLASS_NAMES = [
    "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"
]

st.title("Skin Lesion Image Classifier")
st.write("Upload a dermoscopic skin-lesion image to view the model's predicted class.")

st.warning(
    "Research/educational demonstration only. This model is not a medical device, "
    "is not clinically validated, and must not be used to diagnose or make treatment decisions."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.h5 was not found. Upload the trained CNN model to the same "
            "GitHub folder as app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def load_class_names():
    if CLASS_NAMES_PATH.exists():
        with CLASS_NAMES_PATH.open("r", encoding="utf-8") as f:
            names = json.load(f)
        if isinstance(names, dict):
            # Dict keys must be ordered by their numeric index.
            names = [name for name, index in sorted(names.items(), key=lambda item: item[1])]
        if not isinstance(names, list):
            raise ValueError("class_names.json must contain a JSON list or index-to-name object.")
        return [str(name) for name in names]
    return DEFAULT_CLASS_NAMES

uploaded_file = st.file_uploader(
    "Choose a skin lesion image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        model = load_model()
        class_names = load_class_names()

        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        height = int(input_shape[1]) if input_shape[1] is not None else 224
        width = int(input_shape[2]) if input_shape[2] is not None else 224

        st.caption(
            f"Model input size: {width} × {height}. "
            "The image is resized to this size."
        )

        if st.button("Classify image", type="primary"):
            resized = image.resize((width, height))
            image_array = np.asarray(resized, dtype=np.float32)

            # IMPORTANT: This assumes training used pixel scaling to [0, 1].
            # If your notebook used different preprocessing, match it here.
            image_array = image_array / 255.0
            batch = np.expand_dims(image_array, axis=0)

            predictions = model.predict(batch, verbose=0)
            scores = np.asarray(predictions).squeeze()

            if scores.ndim == 0:
                st.error("The model returned a single score. This app expects a multiclass classifier.")
            else:
                predicted_index = int(np.argmax(scores))
                if len(class_names) != len(scores):
                    st.error(
                        f"Model returns {len(scores)} classes, but class_names.json "
                        f"contains {len(class_names)} labels. Please correct the label file."
                    )
                else:
                    st.subheader("Model output")
                    st.write(f"**Predicted class:** {class_names[predicted_index]}")
                    st.write(f"**Model score:** {float(scores[predicted_index]) * 100:.2f}%")
                    st.caption(
                        "This score is the model's output, not a calibrated probability "
                        "or a measure of diagnostic certainty."
                    )

                    st.write("Scores for each class")
                    for name, score in zip(class_names, scores):
                        st.write(f"{name}: {float(score) * 100:.2f}%")
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
        st.info(
            "Check that model.h5 is a trained Keras image-classification model, "
            "class_names.json matches its output order, and preprocessing matches training."
        )
else:
    st.info("Upload a JPG, JPEG, or PNG image to get started.")
