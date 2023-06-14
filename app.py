import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import random
from Dataset import test_transform as transform_norm
from model import VGG, cfgs, make_layers, selfDefineVgg

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = 'Model\RGB\Main-model=2023-06-13_00-13-47-0.9620.pth'

model = selfDefineVgg(cfgs, 'A', True, 11)
model = model.to(device)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict=state_dict)

upload= st.file_uploader('Insert image for classification', type=['jpg',"jpeg"])
c1, c2= st.columns(2)
if upload is not None:
    img= Image.open(upload)
    img_gray = img.convert("L")#Convert to gray scale image
    img_gray = ImageOps.invert(img_gray).convert("RGB")
    img_normalized = transform_norm(img_gray).float()
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to(device)
    with torch.no_grad(): # Predict Image
        model.eval()
        output =model(img_normalized)
        index = output.data.cpu().numpy().argmax()
        classes = ["Ankle Boot", "Bag", "Coat", "Dress", "Hat","Pullover", "Sandal", "Shirt", "Sneaker", "Tshirt_Top", "Trousers"]
        class_name = classes[index]
        st.image(img, caption=class_name)
        col1,col2,col3 = st.columns(3)
        if class_name == "Ankle Boot":
            with col1:
                st.image(r"Dataset\Ankle Boot\1001.jpg")
            with col2:
                st.image(r"Dataset\Ankle Boot\1002.jpg")
            with col3:
                st.image(r"Dataset\Ankle Boot\1003.jpg")
        
        if class_name == "Bag":
            with col1:
                st.image(r"Dataset\Bag\901.jpg")
            with col2:
                st.image(r"Dataset\Bag\902.jpg")
            with col3:
                st.image(r"Dataset\Bag\903.jpg")
        
        if class_name == "Coat":
            with col1:
                st.image(r"Dataset\Coat\801.jpg")
            with col2:
                st.image(r"Dataset\Coat\804.jpg")
            with col3:
                st.image(r"Dataset\Coat\803.jpg")
        
        if class_name == "Dress":
            with col1:
                st.image(r"Dataset\Dress\701.jpg")
            with col2:
                st.image(r"Dataset\Dress\702.jpg")
            with col3:
                st.image(r"Dataset\Dress\703.jpg")
        
        if class_name == "Hat":
            with col1:
                st.image(r"Dataset\Hat\601.jpg")
            with col2:
                st.image(r"Dataset\Hat\602.jpg")
            with col3:
                st.image(r"Dataset\Hat\603.jpg")
        
        if class_name == "Pullover":
            with col1:
                st.image(r"Dataset\Pullover\501.jpeg")
            with col2:
                st.image(r"Dataset\Pullover\502.jpeg")
            with col3:
                st.image(r"Dataset\Pullover\503.jpeg")
        
        if class_name == "Sandal":
            with col1:
                st.image(r"Dataset\Pullover\402.jpg")
            with col2:
                st.image(r"Dataset\Pullover\404.jpg")
            with col3:
                st.image(r"Dataset\Pullover\405.jpg")
                
        if class_name == "Shirt":
            with col1:
                st.image(r"Dataset\Shirt\301.jpg")
            with col2:
                st.image(r"Dataset\Shirt\302.jpg")
            with col3:
                st.image(r"Dataset\Shirt\303.jpg")
                
        if class_name == "Sneaker":
            with col1:
                st.image(r"Dataset\Sneaker\203.jpg")
            with col2:
                st.image(r"Dataset\Sneaker\206.jpg")
            with col3:
                st.image(r"Dataset\Sneaker\208.jpg")
        if class_name == "Trousers":
            with col1:
                st.image(r"Dataset\Trousers\101.jpg")
            with col2:
                st.image(r"Dataset\Trousers\102.jpg")
            with col3:
                st.image(r"Dataset\Trousers\103.jpg")
                
        if class_name == "Tshirt_Top":
            with col1:
                st.image(r"Dataset\Tshirt_Top\1.jpg")
            with col2:
                st.image(r"Dataset\Tshirt_Top\2.jpg")
            with col3:
                st.image(r"Dataset\Tshirt_Top\3.jpg")