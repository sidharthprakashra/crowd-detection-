# crowd-detection-
Crowd Density Forecasting System using Deep Learning &
YOLOv8
Project Overview
This project is an AI-powered Crowd Density Forecasting System that predicts approximate crowd density at different
locations based on Location, Day Type, and Time Slot. The system combines Computer Vision (YOLOv8), Deep
Learning (PyTorch), and metadata-based forecasting.
Features
• YOLOv8-based crowd detection
• Automated dataset generation from images
• Deep learning crowd forecasting model
• Crowd density classification (LOW, MEDIUM, HIGH)
• Multi-location support
• Time-aware prediction system
Tech Stack
• Python
• PyTorch
• YOLOv8
• OpenCV
• Pandas
• Scikit-learn
• Joblib
Project Workflow
1. Collect images for multiple locations and time slots.
2. Generate crowd dataset using YOLOv8.
3. Train deep learning forecasting model.
4. Save trained weights and encoders.
5. Predict future crowd density using metadata inputs.
Dataset Structure
training_images/
■■■ normal_days/
■■■ exam_special_days/
■■■ event_days/
Project Structure
crowd_forecast/
│
├── training_images/
│
│   ├── normal_days/
│   │   ├── canteen/
│   │   ├── gate1/
│   │   ├── gate2/
│   │   ├── gate3/
│   │   ├── ground/
│   │   └── open_auditorium/
│
│   ├── exam_special_days/
│   └── event_days/
│
├── artifacts/
│   ├── best_model.pt
│   ├── day_encoder.pkl
│   ├── location_encoder.pkl
│   └── time_encoder.pkl
│
├── generate_dataset.py
├── train_model.py
├── predict.py
├── crowd_dataset.csv
└── README.md
Training
Run:
python train_model.py
The model learns relationships between:
(Location + Day Type + Time Slot) → Crowd Count
Prediction
Run:
python predict.py
Example prediction output:
Estimated Crowd Count: 84
Crowd Density Level: HIGH
Model Architecture
Embedding Layers → Fully Connected Layers → Crowd Count Prediction
Future Improvements
• Real-time CCTV integration
• Streamlit dashboard
• Heatmap visualization
• LSTM/Transformer forecasting
• Cloud deployment
Conclusion
This project demonstrates the integration of Computer Vision and Deep Learning for intelligent crowd forecasting
applications.
Crowd Density Levels
Crowd Count Density Level
0 - 20 LOW
21 - 60 MEDIUM
61+ HIGH
