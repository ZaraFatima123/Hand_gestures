import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.title("Hand Gesture Detection Demo")
st.write("This demo detects your hand gestures in real-time.")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

st_frame = st.image([])

# Utility functions
def get_angle(a, b, c):
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    return angle

def get_distance(a, b):
    x1, y1 = a
    x2, y2 = b
    return np.hypot(x2 - x1, y2 - y1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    gesture_label = "No gesture detected"

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmark_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

        if len(landmark_list) >= 21:
            # Calculate angles for fingers
            index_angle = get_angle(landmark_list[5], landmark_list[6], landmark_list[8])
            middle_angle = get_angle(landmark_list[9], landmark_list[10], landmark_list[12])
            ring_angle = get_angle(landmark_list[13], landmark_list[14], landmark_list[16])
            pinky_angle = get_angle(landmark_list[17], landmark_list[18], landmark_list[20])
            thumb_index_dist = get_distance(landmark_list[4], landmark_list[8])

            # Click gesture: only index finger bent
            if index_angle < 50 and middle_angle > 90 and ring_angle > 90 and pinky_angle > 90:
                gesture_label = "Click Gesture (Index Finger Bend)"

            # You can keep other gestures as needed
            elif index_angle < 50 and middle_angle < 50 and thumb_index_dist < 0.05:
                gesture_label = "Screenshot Gesture"

    cv2.putText(frame, gesture_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    st_frame.image(frame, channels="BGR")
