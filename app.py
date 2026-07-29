import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Tiêu đề
st.title("Vietnamese Food AI")
st.write("Nhận diện món ăn Việt Nam bằng AI")

# Load model
model = tf.keras.models.load_model("food_model.keras")

# Tên 30 món ăn
class_names = [
    "Banh beo",
    "Banh bot loc",
    "Banh canh",
    "Banh chung",
    "Banh cuon",
    "Banh khot",
    "Banh mi",
    "Banh trang tron",
    "Banh xeo",
    "Bo kho",
    "Bun bo Hue",
    "Bun dau mam tom",
    "Bun mam",
    "Bun rieu",
    "Bun thit nuong",
    "Ca kho to",
    "Canh chua",
    "Cao lau",
    "Com tam",
    "Goi cuon",
    "Hu tieu",
    "Mi Quang",
    "Nem chua",
    "Pho",
    "Xoi",
    "Che",
    "Cha ca",
    "Com chien",
    "Lau",
    "Sup cua"
]

# Upload ảnh
uploaded_file = st.file_uploader(
    "Chọn ảnh món ăn",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đã tải lên", use_container_width=True)

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    index = np.argmax(prediction)

    st.success(f"Kết quả dự đoán: {class_names[index]}")
