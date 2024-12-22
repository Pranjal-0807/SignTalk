import cv2
import mediapipe as mp
import pyttsx3
import numpy as np

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak_word(word):
    """Speaks the given word using pyttsx3."""
    engine.say(word)
    engine.runAndWait()

# Initialize MediaPipe Hand Detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Define the hand signs for each word
def get_hand_sign(landmarks):
    """Recognize the ASL hand sign based on landmarks."""
    # Thumb tip = 4, Index tip = 8, Middle tip = 12, Ring tip = 16, Pinky tip = 20
    
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Example ASL Gestures (right hand):
    if thumb_tip[1] < index_tip[1] and index_tip[1] < middle_tip[1] and middle_tip[1] < ring_tip[1] and ring_tip[1] < pinky_tip[1]:
        return "Hello"
    elif index_tip[1] < thumb_tip[1] and middle_tip[1] < thumb_tip[1] and ring_tip[1] < thumb_tip[1] and pinky_tip[1] < thumb_tip[1]:
        return "Good"
    elif thumb_tip[0] < index_tip[0] < middle_tip[0] < ring_tip[0] < pinky_tip[0]:
        return "Morning"
    elif index_tip[1] < thumb_tip[1] and middle_tip[1] < index_tip[1] and ring_tip[1] > middle_tip[1] and pinky_tip[1] > ring_tip[1]:
        return "Sir"
    elif thumb_tip[1] > index_tip[1] > middle_tip[1] > ring_tip[1] > pinky_tip[1]:
        return "Madam"
    elif thumb_tip[1] > index_tip[1] and middle_tip[1] > ring_tip[1] and pinky_tip[1] > ring_tip[1] and thumb_tip[0] < index_tip[0]:
        return "How"
    elif pinky_tip[1] < ring_tip[1] and ring_tip[1] < middle_tip[1] and middle_tip[1] < index_tip[1] and index_tip[1] < thumb_tip[1]:
        return "Are"
    elif index_tip[1] < thumb_tip[1] and middle_tip[1] < index_tip[1] and ring_tip[1] < middle_tip[1] and pinky_tip[1] < ring_tip[1]:
        return "You"
    elif thumb_tip[1] < index_tip[1] and pinky_tip[1] > ring_tip[1] and ring_tip[1] > middle_tip[1] and middle_tip[1] > index_tip[1]:
        return "My"
    elif index_tip[0] < middle_tip[0] < ring_tip[0] and pinky_tip[1] > thumb_tip[1]:
        return "Name"
    elif thumb_tip[0] > pinky_tip[0] and middle_tip[1] < ring_tip[1]:
        return "Is"
    elif thumb_tip[1] > pinky_tip[1] and index_tip[1] < middle_tip[1] < ring_tip[1]:
        return "Pranjal"
    elif pinky_tip[1] < index_tip[1] and thumb_tip[0] < index_tip[0]:
        return "Agarwal"

    return None

def process_frame(frame):
    """Process a single frame to detect hand gestures."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            return get_hand_sign(landmarks), frame

    return None, frame

def main():
    """Main function to run the hand gesture recognition and speech system."""
    cap = cv2.VideoCapture(0)

    print("Starting hand gesture recognition. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        word, processed_frame = process_frame(frame)

        if word:
            print(f"Recognized Word: {word}")
            speak_word(word)

        cv2.imshow('Hand Gesture Recognition', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# Working Part => Hello Good Sir 
# import cv2
# import mediapipe as mp
# import pyttsx3
# import numpy as np

# # Initialize text-to-speech engine
# engine = pyttsx3.init()
# def speak_word(word):
#     """Speaks the given word using pyttsx3."""
#     engine.say(word)
#     engine.runAndWait()

# # Initialize MediaPipe Hand Detector
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# # Define the hand signs for each word
# def get_hand_sign(landmarks):
#     """Recognize the ASL hand sign based on landmarks."""
#     # Thumb tip = 4, Index tip = 8, Middle tip = 12, Ring tip = 16, Pinky tip = 20
    
#     thumb_tip = landmarks[4]
#     index_tip = landmarks[8]
#     middle_tip = landmarks[12]
#     ring_tip = landmarks[16]
#     pinky_tip = landmarks[20]

#     # Example ASL Gestures (right hand):
#     if thumb_tip[1] < index_tip[1] and index_tip[1] < middle_tip[1] and middle_tip[1] < ring_tip[1] and ring_tip[1] < pinky_tip[1]:
#         return "Hello"
#     elif index_tip[1] < thumb_tip[1] and middle_tip[1] < thumb_tip[1] and ring_tip[1] < thumb_tip[1] and pinky_tip[1] < thumb_tip[1]:
#         return "Good"
#     elif thumb_tip[0] < index_tip[0] < middle_tip[0] < ring_tip[0] < pinky_tip[0]:
#         return "Morning"
#     elif index_tip[1] < thumb_tip[1] and middle_tip[1] < index_tip[1] and ring_tip[1] > middle_tip[1] and pinky_tip[1] > ring_tip[1]:
#         return "Sir"
#     elif thumb_tip[1] > index_tip[1] > middle_tip[1] > ring_tip[1] > pinky_tip[1]:
#         return "Madam"
#     elif thumb_tip[1] > index_tip[1] and middle_tip[1] > ring_tip[1] and pinky_tip[1] > ring_tip[1] and thumb_tip[0] < index_tip[0]:
#         return "How"
#     elif pinky_tip[1] < ring_tip[1] and ring_tip[1] < middle_tip[1] and middle_tip[1] < index_tip[1] and index_tip[1] < thumb_tip[1]:
#         return "Are"
#     elif index_tip[1] < thumb_tip[1] and middle_tip[1] < index_tip[1] and ring_tip[1] < middle_tip[1] and pinky_tip[1] < ring_tip[1]:
#         return "You"
    
#     return None

# def process_frame(frame):
#     """Process a single frame to detect hand gestures."""
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
#             return get_hand_sign(landmarks), frame

#     return None, frame

# def main():
#     """Main function to run the hand gesture recognition and speech system."""
#     cap = cv2.VideoCapture(0)

#     print("Starting hand gesture recognition. Press 'q' to quit.")

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame.")
#             break

#         word, processed_frame = process_frame(frame)

#         if word:
#             print(f"Recognized Word: {word}")
#             speak_word(word)

#         cv2.imshow('Hand Gesture Recognition', processed_frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
