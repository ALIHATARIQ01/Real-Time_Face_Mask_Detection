# Real-Time Face Mask Detection (Tkinter Desktop App)

## 📌 Overview
This project detects whether a person is wearing a mask or not in real-time using:
- TensorFlow (MobileNetV2)
- OpenCV for face detection
- Tkinter for GUI deployment

## 🧰 Requirements
Install required packages:

```
pip install -r requirements.txt
```
## 📥 Download Dataset
You can download a pre-collected face mask dataset (e.g., Kaggle Face Mask Dataset) using the following link:
Download Dataset from Kaggle

After downloading, extract it and place it in the data/ folder as:
```
data/
├── with_mask/
├── without_mask/
```

## 🚀 How to Run

1. Place your dataset inside the `data/` folder:
```
data/
├── with_mask/
├── without_mask/
```

2. Run data preprocessing & training:
```
python data_preprocessing.py
python train_model.py
```

3. Launch GUI:
```
python app_gui.py
```

## ✅ Author
Aliha Tariq
Computer Science Enthusiast | Passionate about AI & Image Processing
🔗 [GitHub](https://github.com/ALIHATARIQ01)

