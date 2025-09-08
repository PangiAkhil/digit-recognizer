import streamlit as st
from streamlit_drawable_canvas import st_canvas
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ----------------- 🧠 Page Setup -----------------
st.set_page_config(page_title="Digit Recognizer", layout="wide")

st.title("✍️ Handwritten Digit Recognition App")
st.write("Welcome! Draw a digit (0-9) or upload an image, and this app will predict it using a deep learning model trained on the MNIST dataset.")

# ----------------- 🎨 Sidebar Settings -----------------
with st.sidebar:
    st.header("🛠️ Drawing Tools")
    
    # 🌙 Dark mode toggle
    dark_mode = st.toggle("🌙 Enable Dark Mode")
    
    if dark_mode:
        stroke_color = "#FFFFFF"
        bg_color = "#000000"
    else:
        stroke_color = "#000000"
        bg_color = "#FFFFFF"
    
    drawing_mode = st.selectbox("Choose drawing tool:", ("freedraw", "line", "rect", "circle", "transform"))
    stroke_width = st.slider("Stroke width:", 1, 50, 20)
    realtime_update = st.checkbox("Update in realtime", True)

    # 🧽 Clear canvas
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = "canvas_0"
    if st.button("🧽 Clear Canvas"):
        current_key = int(st.session_state.canvas_key.split("_")[1])
        st.session_state.canvas_key = f"canvas_{current_key + 1}"

    st.header("📤 Upload Digit Image")
    uploaded_file = st.file_uploader("Upload an image (28x28 recommended)", type=["png", "jpg", "jpeg"])

# ----------------- 🔄 Load Model -----------------
@st.cache_resource
def load_mnist_model():
    return load_model("digit_recog.keras")

model = load_mnist_model()

# ----------------- 🔧 Preprocessing -----------------
def preprocess_image(image):
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    elif image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = 255 - image
    _, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        digit = image[y:y+h, x:x+w]
    else:
        digit = image

    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.pad(digit, ((4, 4), (4, 4)), mode='constant', constant_values=0)
    normalized = padded / 255.0
    return normalized.reshape(1, 28, 28, 1)

# ----------------- 📊 Probability Plot -----------------
def plot_probabilities(prediction):
    fig, ax = plt.subplots(figsize=(6, 3))
    digits = list(range(10))
    ax.bar(digits, prediction[0], color='skyblue')
    ax.set_xticks(digits)
    ax.set_ylim([0, 1])
    ax.set_title("Prediction Probabilities")
    ax.set_xlabel("Digit")
    ax.set_ylabel("Confidence")
    st.pyplot(fig)

# ----------------- 🎨 Drawing or Upload Input -----------------
col1, col2 = st.columns([2, 1])
img_for_prediction = None
raw_image_display = None

with col1:
    st.subheader("🖌️ Draw a Digit")

    canvas_result = st_canvas(
        fill_color="rgba(255,165,0,0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        update_streamlit=realtime_update,
        height=280,
        width=280,
        drawing_mode=drawing_mode,
        key=st.session_state.canvas_key
    )

    # Upload image
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((280, 280))  # Resize for consistency
        img_array = np.array(image)
        img_for_prediction = preprocess_image(img_array)
        raw_image_display = image
    elif canvas_result.image_data is not None:
        img_array = canvas_result.image_data.astype("uint8")
        img_for_prediction = preprocess_image(img_array)
        raw_image_display = img_array

    # Show preview if image is ready
    if raw_image_display is not None:
        st.image(raw_image_display, caption="🖼️ Image (Processed)", width=150)

# ----------------- 🔍 Prediction -----------------
with col2:
    st.subheader("🔍 Prediction")

    if img_for_prediction is not None:
        if st.button("🔮 Predict Digit"):
            prediction = model.predict(img_for_prediction)
            predicted_digit = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            st.success(f"✅ Predicted Digit: {predicted_digit}")
            st.write(f"📊 Confidence: {confidence:.2%}")

            st.subheader("📈 Probability for each digit")
            plot_probabilities(prediction)

            with st.expander("See raw prediction values"):
                for i, prob in enumerate(prediction[0]):
                    st.write(f"{i}: {prob:.4f}")
    else:
        st.info("Draw or upload a digit image to enable prediction.")

# ----------------- 👤 Footer -----------------
st.markdown("---")
st.write("Made with ❤️ by **PANGI AKHIL** under the guidance of **Saxon K Sha** at **Innomatics Research Labs**.")
