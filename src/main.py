import cv2
import numpy as np
import time
import mediapipe as mp
import math as math
import tkinter as tk

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

#Screen dimensions
screen_width = 1280
screen_height = 720

#For Detecting pinch smoothly
pinch_wait = 1.0 
last_pinch_time = 0

#For detecting horizontal swipe
previous_x = None  

def handle_gesture_controls(hand_landmarks):
    """Handle gestures """
    global selected_song_index, previous_x, swipe_target, currentMiddleIndex
    global last_pinch_time, state, last_played_song_index

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Convert to pixel positions
    x1, y1 = int(thumb_tip.x * screen_width), int(thumb_tip.y * screen_height)
    x2, y2 = int(index_finger_tip.x * screen_width), int(index_finger_tip.y * screen_height)

    #Distance btw thumb and index finger
    pinch_distance = math.hypot(x2 - x1, y2 - y1)

    current_time = time.time()

    #If Pinch is detected
    if pinch_distance < 40: 
        if current_time - last_pinch_time > pinch_wait: 
            print("Pinch is detected.")
              
            last_pinch_time = current_time 
        return  
        # Calculate horizontal swipe movement for scrolling
    if previous_x is None:
        previous_x = x2  # Initialize previous_x if not set
        return

    delta_x = x2 - previous_x  # difference of current x with x-axis of previous frame
    previous_x = x2  #update previous_x with current frame

    # Detect swipe gesture
    if abs(delta_x) > 50:  # Swipe threshold = 50, most optimal for group's hand sizes and smooth gesture
        if delta_x > 0: # Swipe Right
            print("Right Swipe")
        else:  # Swipe Left
	        print("Left Swipe")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
            handle_gesture_controls(hand_landmarks)

    cv2.imshow('A5', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()