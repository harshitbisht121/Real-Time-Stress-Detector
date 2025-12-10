# Real-Time-Stress-Detector
A deep-learning powered facial emotion and stress detection system that uses a Convolutional Neural Network (CNN) to analyze facial expressions and classify emotional states in real-time. The project includes a training pipeline, a saved model (model.h5), and a live webcam-based detection interface built using OpenCV and Eel for a lightweight UI.
🚀 Features

🧠 CNN-based emotion & stress detection

🎥 Real-time detection using webcam

🖼️ Automatic face detection using Haar Cascade

🧪 Image preprocessing & augmentation

🗂️ Custom dataset training using TensorFlow/Keras

💡 Simple and lightweight UI via Eel

📦 Includes model.h5 for direct usage without retraining

🛠️ Technologies Used

Python 3

TensorFlow / Keras

OpenCV

Eel

NumPy

Pandas

Matplotlib

Seaborn

📁 Project Structure
Real-Time-Stress-Detector/
│
├── images/                         # Dataset (train/validation)
│
├── main.py                         # Real-time detection application
├── model.h5                        # Trained CNN model
├── haarcascade_frontalface_default.xml  # Face detection classifier
│
├── emotion-classification-cnn.ipynb     # Model training notebook
│
├── web/                            # Eel frontend files (HTML/CSS/JS)
│
└── README.md

🧩 How It Works

The CNN model is trained on facial expression images (48×48 grayscale).

The system detects the face using OpenCV Haar Cascade.

The detected face is preprocessed and passed to the model.

The model predicts one of the emotion labels:

Angry

Disgust

Fear

Happy

Neutral

Sad

Surprise

Based on these emotions, the system identifies stress-related states.

▶️ Running the Project
1. Install dependencies
pip install tensorflow opencv-python eel numpy pandas matplotlib seaborn

2. Run the application
python main.py


Your webcam will open, and real-time detection will begin.

🧠 Training the Model (Optional)

You can retrain or modify the CNN using:

emotion-classification-cnn.ipynb


This notebook includes:

Data loading

Data augmentation

Model architecture

Training callbacks

Model saving (model.h5)
