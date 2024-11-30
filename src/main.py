import cv2
import numpy as np
import time
import mediapipe as mp
import math as math
import tkinter as tk

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    #Flip the image horizontally
    image = cv2.flip(image, 1)
    results = hands.process(image)

    #Draw the hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('A5', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()