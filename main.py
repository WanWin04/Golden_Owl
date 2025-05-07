import streamlit as st
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader

# Set page config
st.set_page_config(page_title="Dog vs Cat Classifier", page_icon="🐾", layout="centered")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)
    checkpoint = torch.load("./model/checkpoint_epoch_10.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

model = load_model()

def RandomImagePrediction(uploaded_file):
    try:
        img_array = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        return f"Lỗi khi đọc ảnh: {e}"
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = data_transforms(img_array).unsqueeze(0)
    load = DataLoader(img)

    for x in load:
        x = x.to(device)
        pred = model(x)
        _, preds = torch.max(pred, 1)
        return "🐶 Chó" if preds[0] == 1 else "🐱 Mèo"
    return "Lỗi không xác định"

# Main UI
st.markdown(
    "<h1 style='text-align: center; color: #4A4A4A;'>📸 Phân loại Hình ảnh: Chó hay Mèo?</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Tải lên một hình ảnh và chúng tôi sẽ dự đoán đó là chó 🐶 hay mèo 🐱.</p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📁 Chọn một ảnh (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
    
    with col2:
        uploaded_file.seek(0)
        prediction = RandomImagePrediction(uploaded_file)
        st.markdown(
            f"""
            <div style='padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;'>
                <h3 style='color: #333;'>👉 Dự đoán:</h3>
                <h2 style='color: #FF4B4B;'>{prediction}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; font-size: 14px; color: gray;'>Mô hình học sâu sử dụng ResNet50 và huấn luyện trước với PyTorch.</p>",
        unsafe_allow_html=True
    )
