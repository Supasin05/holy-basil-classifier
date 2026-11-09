import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

# ตั้งค่าหน้าเว็บ
st.set_page_config(
   page_title="ระบบตรวจโรคใบกะเพราอัจฉริยะ",
   page_icon="🌱",
   layout="centered"
)

# โหลดโมเดล
BEST_PATH = '/content/best_basil_mobilenetv2.pt'
CLASSES = ['bad', 'good']
IMG_SIZE = 224

@st.cache_resource
def load_model():
   # โครงสร้าง MobileNetV2
   from torchvision import models
   model = models.mobilenet_v2(weights=None)
   model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.last_channel, 2))
   if os.path.exists(BEST_PATH):
       ckpt = torch.load(BEST_PATH, map_location='cpu')
       model.load_state_dict(ckpt['state_dict'])
   model.eval()
   return model

try:
   model = load_model()
   cam = GradCAM(model=model, target_layers=[model.features[-1]])
except Exception as e:
   st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")

# การแปลงภาพตามที่โมเดลต้องการ
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
eval_tf = transforms.Compose([
   transforms.Resize((IMG_SIZE, IMG_SIZE)),
   transforms.ToTensor(),
   transforms.Normalize(mean, std)
])

# หน้าตา UI ของเว็บ
st.title("🌱 ระบบตรวจสภาพใบกะเพราอัจฉริยะ (Holy Basil Leaf Quality)")
st.write("อัปโหลดรูปภาพใบกะเพราของคุณ เพื่อให้ระบบวิเคราะห์สภาพของใบพร้อมแสดงจุดที่พบปัญหา")

uploaded_file = st.file_uploader("เลือกรูปภาพใบกะเพรา (.jpg, .jpeg, .png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
   # แสดงภาพต้นฉบับ
   image = Image.open(uploaded_file).convert('RGB')

   col1, col2 = st.columns(2)
   with col1:
       st.subheader("📷 ภาพที่อัปโหลด")
       st.image(image, use_container_width=True)

   # ประมวลผลและทำนายผล
   with st.spinner('กำลังวิเคราะห์โครงสร้างใบ...'):
       # สำหรับทำนาย
       tensor_img = eval_tf(image).unsqueeze(0)
       with torch.no_grad():
           outputs = model(tensor_img)
           probs = torch.softmax(outputs, dim=1).numpy()[0]
           pred_idx = int(probs.argmax())
           confidence = probs[pred_idx]

       # คำนวณ Grad-CAM
       # ปรับสเกลภาพต้นฉบับสำหรับแสดงผล overlay
       resized_img = image.resize((IMG_SIZE, IMG_SIZE))
       rgb_img = np.asarray(resized_img, dtype=np.float32) / 255.0

       # หา Heatmap จุดเด่นในการจำแนก
       grayscale_cam = cam(input_tensor=tensor_img, targets=None)[0]
       cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

   with col2:
       st.subheader("🔍 พื้นที่วิเคราะห์ (Grad-CAM)")
       st.image(cam_image, use_container_width=True, caption="บริเวณสีแดงคือจุดที่โมเดลใช้ในการตัดสินใจ")

   # สรุปผลการวิเคราะห์
   st.markdown("--- ")
   st.subheader("📊 สรุปผลการวินิจฉัย")

   status = CLASSES[pred_idx]
   if status == 'good':
       st.success(f"**ผลลัพธ์: ใบกะเพราสมบูรณ์ดี (Healthy)** (ความมั่นใจ {confidence:.2%})")
       st.balloons()
       st.info("💡 **คำแนะนำ:** ใบกะเพราของคุณมีสภาพสมบูรณ์และพร้อมสำหรับการนำไปบริโภคหรือประกอบอาหารได้อย่างปลอดภัย หมั่นดูแลรดน้ำและให้ปุ๋ยสม่ำเสมอครับ")
   else:
       st.error(f"**ผลลัพธ์: ใบกะเพราชำรุดหรือมีโรคใบเสีย (Unhealthy / diseased)** (ความมั่นใจ {confidence:.2%})")
       st.warning("💡 **คำแนะนำและแนวทางแก้ไข:**\n"
                  "1. **ตรวจสอบความเสียหาย:** ตรวจดูว่าแผลเกิดจากเชื้อรา เพลี้ย หรือขาดสารอาหาร (สังเกตบริเวณสีแดงเด่นชัดในภาพ Grad-CAM)\n"
                  "2. **ตัดกิ่ง/ใบที่เสียหายออก:** แนะนำให้ตัดใบที่มีปัญหาทิ้งทันที เพื่อลดการแพร่กระจายของเชื้อโรคไปยังส่วนอื่น\n"
                  "3. **การรักษา:** หลีกเลี่ยงการรดน้ำที่ใบโดยตรงในช่วงเย็นเพื่อลดความชื้นสะสม และพิจารณาใช้สารชีวภัณฑ์ในการกำจัดโรคพืช")
