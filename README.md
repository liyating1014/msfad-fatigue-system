
# MS-FAD: See Through Your Fatigue Like Your Eyes

## 🔍 Overview

MS-FAD (Multi-Scale Fatigue Attention Detection) is a lightweight and efficient driver fatigue detection framework that combines multi-scale convolutional attention (MSCA), bidirectional feature pyramid network (BiFPN), and a streamlined detection head (ScHead). It achieves state-of-the-art accuracy with significantly reduced computational cost, making it ideal for real-time deployment in embedded and web-based scenarios.


## 📈 Highlights

 Lightweight: Only 4.9 GFLOPs and 1.68M parameters
 Accuracy: mAP@0.5 = **0.897**, outperforming YOLOv8n baseline by **+3%**
 Web-Ready: Integrated with a full-stack fatigue detection system (Next.js + Koa + MySQL)
 Modular Design: Easy to extend or deploy on custom datasets or platforms

## 🧠 Architecture

MS-FAD consists of:
- **Backbone** with MSCA (Multi-Scale Convolutional Attention)
- **Neck** based on BiFPN for efficient multi-scale fusion
- **ScHead** for lightweight detection with SCConv
- **Web Interface** for real-time fatigue alerting

## 📦 Coming Soon

- [ ] PyTorch implementation with training/inference code
- [ ] Pretrained models on Huawei & YawDD datasets
- [ ] Sample dataset & annotation format
- [ ] Web frontend and backend source code
- [ ] Docker deployment template

