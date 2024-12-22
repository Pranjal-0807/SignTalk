from function import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import cv2
import numpy as np
import mediapipe as mp
from gtts import gTTS
from playsound import playsound

# Load the model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Define colors for visualization
colors = [(245, 117, 16) for _ in range(20)]

# Variables for detection and visualization
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8
last_action = None  # Tracks the last detected action

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

# Using MediaPipe Hands model
with mp_hands.Hands(
    
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        # Read video feed
        ret, frame = cap.read()
        if not ret:
            break

        # Define the active region and crop
        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)

        # Perform detections
        image, results = mediapipe_detection(cropframe, hands)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep the last 30 keypoints

        try:
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_action = actions[np.argmax(res)]
                confidence = res[np.argmax(res)]

                # Update only for new actions above the threshold
                if confidence > threshold and predicted_action != last_action:
                    sentence.append(predicted_action)
                    accuracy.append(f"{confidence * 100:.2f}")
                    last_action = predicted_action

                    # Speak the detected action using gTTS
                    # mytext = f"The detected action is {predicted_action}"
                    mytext = f"{predicted_action}"
                    language = 'en'
                    myobj = gTTS(text=mytext, lang=language, slow=False)
                    myobj.save("action.mp3")
                    playsound("action.mp3")

                # Limit to the latest action
                if len(sentence) > 1:
                    sentence = sentence[-1:]
                    accuracy = accuracy[-1:]

        except Exception as e:
            pass

        # Display the output on the frame
        cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
        output_text = f"Output: {' '.join(sentence)} {''.join(accuracy)}"
        cv2.putText(frame, output_text, (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('OpenCV Feed', frame)

        # Break on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
