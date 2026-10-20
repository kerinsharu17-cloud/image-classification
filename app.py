import os
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="CNN Image Classifier", page_icon="🐱", layout="wide")
os.makedirs("outputs", exist_ok=True)
MODEL_PATH = "outputs/cat_dog_cnn.keras"

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        import tensorflow as tf
        return tf.keras.models.load_model(path)
    return None

def train(epochs, limit):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.datasets import cifar10
    import matplotlib.pyplot as plt

    (xt, yt), (xv, yv) = cifar10.load_data()
    yt, yv = yt.flatten(), yv.flatten()
    mt, mv = np.isin(yt, [3, 5]), np.isin(yv, [3, 5])
    xt, yt = xt[mt][:limit], yt[mt][:limit]
    xv, yv = xv[mv][:1000], yv[mv][:1000]
    yt, yv = (yt == 5).astype("float32"), (yv == 5).astype("float32")
    xt, xv = xt.astype("float32") / 255.0, xv.astype("float32") / 255.0

    model = models.Sequential([
        layers.Input(shape=(32,32,3)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(xt, yt, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
    loss, acc = model.evaluate(xv, yv, verbose=0)
    model.save(MODEL_PATH)
    for key, title, ylabel, filename in [
        ("accuracy", "CNN Accuracy", "Accuracy", "accuracy.png"),
        ("loss", "CNN Loss", "Loss", "loss.png")
    ]:
        fig, ax = plt.subplots()
        ax.plot(history.history[key], label="Training " + ylabel)
        ax.plot(history.history["val_" + key], label="Validation " + ylabel)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.legend()
        fig.tight_layout(); fig.savefig("outputs/" + filename); plt.close(fig)
    return acc, loss

st.title("🐱🐶 CNN Image Classification Dashboard")
st.write("Upload a JPG or PNG image and predict whether it looks like a cat or dog.")
st.caption("Local Streamlit dashboard. The first training run needs internet to download CIFAR-10.")

with st.sidebar:
    st.header("Model training")
    epochs = st.slider("Epochs", 1, 15, 5)
    limit = st.selectbox("Training images", [2000, 5000, 8000], index=1)
    clicked = st.button("Train / Retrain CNN", type="primary", use_container_width=True)
    st.write("Model status:", "Trained model found" if os.path.exists(MODEL_PATH) else "Not trained yet")

if clicked:
    try:
        with st.spinner("Downloading data if needed and training the CNN..."):
            acc, loss = train(epochs, limit)
        load_model.clear()
        st.success(f"Training complete. Test accuracy: {acc*100:.2f}% | Test loss: {loss:.4f}")
        st.rerun()
    except Exception as e:
        st.error("Training failed. Check the internet connection and TensorFlow installation.")
        st.code(str(e))
        st.info("See README.md for setup and dataset troubleshooting.")

model = load_model(MODEL_PATH)
col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Upload an image")
    uploaded = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
    image = None
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)
        except Exception as e:
            st.error(f"Unable to open image: {e}")
with col2:
    st.subheader("2. Prediction")
    if model is None:
        st.warning("Train the model first using the button in the sidebar.")
    elif image is not None:
        arr = np.asarray(image.resize((32,32)), dtype=np.float32) / 255.0
        p_dog = float(model.predict(arr[None, ...], verbose=0)[0][0])
        label = "Dog" if p_dog >= 0.5 else "Cat"
        confidence = p_dog if label == "Dog" else 1-p_dog
        st.metric("Predicted class", label)
        st.progress(float(np.clip(confidence, 0, 1)))
        st.write(f"Confidence: **{confidence*100:.2f}%**")
        st.write(f"Dog score: {p_dog*100:.2f}%")
        st.write(f"Cat score: {(1-p_dog)*100:.2f}%")
        st.caption("Educational demo trained on small 32×32 CIFAR-10 images; real-world images may be misclassified.")
st.divider()
st.subheader("Training graphs")
g1, g2 = st.columns(2)
with g1:
    if os.path.exists("outputs/accuracy.png"): st.image("outputs/accuracy.png", use_container_width=True)
    else: st.caption("Accuracy graph appears after training.")
with g2:
    if os.path.exists("outputs/loss.png"): st.image("outputs/loss.png", use_container_width=True)
    else: st.caption("Loss graph appears after training.")
with st.expander("About this project"):
    st.write("CNN uses convolution and pooling layers to learn image patterns. The final sigmoid layer predicts Cat or Dog. Streamlit provides the local browser dashboard.")