# app.py
import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
import cv2
import random

st.set_page_config(page_title="Virtual Mouse", layout="wide")
st.title("🖐 Virtual Mouse Simulator with Gestures")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Canvas for virtual cursor
canvas_width, canvas_height = 640, 480
canvas = Image.new("RGB", (canvas_width, canvas_height), (30, 30, 30))

# Camera input
img_file_buffer = st.camera_input("Show your hand")
if img_file_buffer is not None:
    # Convert to OpenCV image
    image = Image.open(img_file_buffer)
    frame = np.array(image)
    frame = cv2.resize(frame, (canvas_width, canvas_height))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    result = hands.process(frame_rgb)
    draw = ImageDraw.Draw(canvas)
    action = "No Action"

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # Draw hand landmarks
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get finger landmarks
        tips = {
            "thumb": hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
            "index": hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
            "middle": hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            "ring": hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
            "pinky": hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        }

        pips = {
            "index": hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
            "middle": hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
            "ring": hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP],
            "pinky": hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
        }

        # Virtual cursor position (index finger tip)
        x = int(tips["index"].x * canvas_width)
        y = int(tips["index"].y * canvas_height)
        cursor_radius = 15

        # Draw cursor
        draw.ellipse((x-cursor_radius, y-cursor_radius, x+cursor_radius, y+cursor_radius), fill=(255, 0, 0))

        # Detect gestures
        index_bent = tips["index"].y > pips["index"].y
        middle_bent = tips["middle"].y > pips["middle"].y
        ring_bent = tips["ring"].y > pips["ring"].y
        pinky_bent = tips["pinky"].y > pips["pinky"].y

        # Gesture logic
        if index_bent and not middle_bent:
            action = "Left Click"
            draw.ellipse((x-cursor_radius, y-cursor_radius, x+cursor_radius, y+cursor_radius), fill=(0, 255, 0))
        elif middle_bent and not index_bent:
            action = "Right Click"
            draw.ellipse((x-cursor_radius, y-cursor_radius, x+cursor_radius, y+cursor_radius), fill=(0, 0, 255))
        elif index_bent and middle_bent:
            action = "Double Click"
            draw.ellipse((x-cursor_radius, y-cursor_radius, x+cursor_radius, y+cursor_radius), fill=(255, 255, 0))
        elif index_bent and middle_bent and ring_bent and pinky_bent:
            action = "Screenshot"
            label = random.randint(1000, 9999)
            canvas.save(f"screenshot_{label}.png")
            draw.ellipse((x-cursor_radius, y-cursor_radius, x+cursor_radius, y+cursor_radius), fill=(255, 0, 255))

    # Display webcam and virtual mouse canvas
    col1, col2 = st.columns(2)
    col1.image(frame, channels="BGR", caption="Webcam Feed")
    col2.image(canvas, caption=f"Virtual Cursor Simulator: {action}")
