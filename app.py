"""
Vietnamese Food AI - Streamlit Web App
----------------------------------------
Web đơn giản: người dùng upload ảnh món ăn Việt Nam -> AI đoán đó là món gì.

Cách chạy ở máy local:
    pip install streamlit tensorflow pillow numpy
    streamlit run app.py

LƯU Ý: file này cần 2 file được tạo ra từ notebook train:
    - vietnamese_food_mobilenetv2.keras   (model đã train)
    - class_indices.json                  (mapping số -> tên món ăn)
Nhớ để 2 file này cùng thư mục với app.py (hoặc sửa lại đường dẫn MODEL_PATH / CLASS_MAP_PATH bên dưới).
"""

import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------------------------------------------------
# Cấu hình chung
# ---------------------------------------------------------
MODEL_PATH = "vietnamese_food_mobilenetv2.keras"
CLASS_MAP_PATH = "class_indices.json"
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Vietnamese Food AI", page_icon="🍜", layout="centered")


# ---------------------------------------------------------
# Load model + class mapping (dùng cache để không load lại mỗi lần người dùng tương tác)
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        idx_to_class_raw = json.load(f)
    # JSON luôn lưu key dạng string, cần convert lại về int để tra cứu đúng
    idx_to_class = {int(k): v for k, v in idx_to_class_raw.items()}
    return model, idx_to_class


def predict(image: Image.Image, model, idx_to_class, top_k: int = 3):
    """Tiền xử lý ảnh và trả về top_k dự đoán (tên món, xác suất)."""
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # PHẢI giống hệt cách tiền xử lý lúc train

    preds = model.predict(img_array, verbose=0)[0]
    top_indices = preds.argsort()[-top_k:][::-1]
    return [(idx_to_class[i], float(preds[i])) for i in top_indices]


# ---------------------------------------------------------
# Giao diện
# ---------------------------------------------------------
st.title("🍜 Vietnamese Food AI")
st.write(
    "Upload ảnh một món ăn Việt Nam, AI sẽ đoán xem đó là món gì trong 30 món "
    "(Phở, Bún bò Huế, Bánh mì, Gỏi cuốn, ...)."
)

try:
    model, idx_to_class = load_model_and_classes()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(
        "Không tìm thấy hoặc không load được model. "
        "Hãy chắc chắn 2 file `vietnamese_food_mobilenetv2.keras` và `class_indices.json` "
        "nằm cùng thư mục với app.py.\n\nChi tiết lỗi: " + str(e)
    )

uploaded_file = st.file_uploader("Chọn ảnh món ăn (jpg, png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh đã upload", use_container_width=True)

    with st.spinner("AI đang phân tích ảnh..."):
        results = predict(image, model, idx_to_class, top_k=3)

    best_name, best_prob = results[0]
    st.success(f"**Dự đoán: {best_name}** ({best_prob * 100:.1f}% chắc chắn)")

    st.write("Top 3 khả năng:")
    for name, prob in results:
        st.write(f"- {name}: {prob * 100:.1f}%")
        st.progress(min(prob, 1.0))

st.markdown("---")
st.caption(
    "Model: MobileNetV2 (Transfer Learning) huấn luyện trên dataset "
    "[Vietnamese Foods (Kaggle)](https://www.kaggle.com/datasets/quandang/vietnamese-foods)."
)
