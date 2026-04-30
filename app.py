import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from utils import predict_image

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # kernel size = 2, stride = 2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) ,

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4*4*128, 256),
            nn.ReLU(),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # flattening
        x = self.fc_layers(x)

        return x
    
# Load model
model = CNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Custom CSS for background color
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: ##2E7D32;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


st.markdown(
    """
    <style>
    [data-testid="stHeader"] {
        margin-top: -60px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# 3. Streamlit UI
st.markdown("<h1 style = 'text-align:center; color:orange; margin-bottom:-25px; font-size: 60px;'> Fruit Freshness Detection<br>🍌🍎🍊</br></h1>",unsafe_allow_html=True)
st.markdown(
        "<p style='color: white; margin-top:-35px; text-align:left; font-size:24px;'>👉 " \
        "An image classification model for detecting fresh and rotten apples, bananas, and oranges.<br>⚠️Don’t risk your health with spoiled fruit</br>✨ Ensure freshness before consumption by uploading an image 📸 </p>",
        unsafe_allow_html=True
    )


# Upload section

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((100,100))
    st.image(image, caption="Uploaded Successfully!")

    # Preprocess
    prediction = predict_image(image)

    if "Fresh" in prediction:
       color ="#3CB02E" 
       tip = "✨Fresh fruits are rich in vitamins and safe to eat."  
    else:
       color = "#EB3D3D" 
       tip = "⚠️Rotten fruits may cause stomach issues, avoid eating."  

    st.markdown(
        f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; color: black; font-size:18px;'>Prediction: {prediction}</div>",
         unsafe_allow_html=True
    )

    # Tips box
    st.markdown(
        f"<p style='text-align: left; font-size:18px; color: white; margin-top:6px;'>{tip}</p>",
        unsafe_allow_html=True
    )
    
st.markdown(
    "<p style='text-align: right; color: orange; font-size:16px;'>By Masudur Rahman</p>",
    unsafe_allow_html=True
)


