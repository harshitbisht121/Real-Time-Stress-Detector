from keras.models import load_model
from time import sleep
from keras.preprocessing import image
import cv2
import numpy as np
import eel

eel.init('C:/Users/hbish/Downloads/Real-time Stress Detector/web') 

face_classifier = cv2.CascadeClassifier(
    r'C:/Users/hbish/Downloads/Real-time Stress Detector/haarcascade_frontalface_default.xml'
)
classifier = load_model(
    r'C:/Users/hbish/Downloads/Real-time Stress Detector/model.h5'
)

# IMPORTANT: must match train_set.class_indices order from notebook
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

@eel.expose
def start():
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum(roi_gray) != 0:
                roi = roi_gray.astype('float32') / 255.0     # SAME as training
                roi = np.expand_dims(roi, axis=-1)           # (48,48,1)
                roi = np.expand_dims(roi, axis=0)            # (1,48,48,1)

                prediction = classifier.predict(roi)[0]
                max_prob = float(np.max(prediction))
                emotion_idx = int(np.argmax(prediction))
                emotion = emotion_labels[emotion_idx]

                # DEBUG (optional): see what the model is doing
                # print("Pred:", np.round(prediction, 3), "->", emotion, "max_prob:", max_prob)

                # If the model is unsure, assume Neutral (low stress)
                if max_prob < 0.27:   # you can tune 0.40 to 0.35 or 0.45 later
                    emotion = 'Neutral'
                    stress = 'low stressed'
                else:
                # Stress mapping
                    if emotion in ['Fear', 'Angry', 'Sad']:
                        stress = 'highly stressed'
                    elif emotion in ['Disgust', 'Neutral']:
                        stress = 'low stressed'
                    elif emotion in ['Happy']:
                        stress = 'not stressed'
                    else:
                        stress = 'uncertain'

                label_text = f"{emotion} - {stress}"
                label_position = (x, y)
                cv2.putText(frame, label_text, label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

eel.start('index.html', port=8080, cmdline_args=['--start--fullscreen', '--browser-starup-dialog'])
