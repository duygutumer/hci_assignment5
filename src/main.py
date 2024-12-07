import cv2
import os
import time
import mediapipe as mp
import math as math
import tkinter as tk

# Suppress TensorFlow and MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

#Screen dimensions
screen_width = 1280
screen_height = 720

#For Detecting pinch smoothly
pinch_wait = 1.0 
last_pinch_time = 0

#For detecting horizontal swipe
previous_x = None  

#State (for album page and music list page)
state = "NORMAL" #first page (albums list) as default
shift_x = 0  # Horizontal shift for smooth transitions
# Album box settings
smallDisplay = (100, 100)

def generate_positions(album_count):
    """Calculate positions of album covers based on shift_x."""
    box_width = screen_width // (album_count + 1)  # Equal spacing
    positions = []
    for i in range(album_count):
        x = (i * box_width) + shift_x  # Add horizontal shift
        y = screen_height // 2 - 150
        positions.append({"pos": (x, y), "size": smallDisplay, "color": (0, 255, 0)})
    return positions

# Album data
albums = [
    {"cover": "./Duman.jpg", "name": "Belki Alisman Lazim", "artist": "Duman", 
     "songs": ["./Sor Bana Pisman miyim.mp3", "./Kufi.mp3", "./Senden Daha Guzel.mp3", "./Kolay Degildir.mp3", "./Yurek.mp3"]},
    {"cover": "./BarisManco.jpg", "name": "Baris Mancho", "artist": "Baris Manco", 
      "songs": ["./Alla Beni Pulla Beni.mp3", "./Arkadasim Esek.mp3", "./Gulpembe.mp3", "./Sari Cizmeli Mehmet Aga.mp3", "./Yaz Dostum.mp3"]},
    {"cover": "./EdSheeran.jpg", "name": "Perfect", "artist": "Ed Sheeran", 
     "songs": ["./Shape of You.mp3", "./Shivers.mp3", "./Photograph.mp3", "./Perfect.mp3", "./Thinking Out Loud.mp3"]},
    {"cover": "./Ottoman.jpg", "name": "Osmanli Marslari", "artist": "Anonim", 
     "songs": ["./Ceddin Deden.mp3", "./Ey Sanli Ordu Ey Sanli Asker.mp3", "./Hucum Marsi.mp3", "./Gafil Ne Bilir.mp3", "./Yelkenler Bicilecek.mp3"]},
    {"cover": "./ZekiMuren.jpg", "name": "Sanat Gunesi", "artist": "Zeki Muren", 
     "songs": ["./Ah bu sarkilarin gozu kor olsun.mp3", "./Gitme Sana Muhtacim.mp3", "./Sorma Ne Haldeyim.mp3", "./Ben Zeki Muren.mp3", "./Seviyorum iste var mi diyecegin.mp3"]},
]

# Load and resize album covers
album_covers = [cv2.imread(album["cover"]) for album in albums]
cover_images_resized = []
for i, cover in enumerate(album_covers):
    if cover is None:
        print(f"Error loading image: {albums[i]['cover']}")
    else:
        cover_images_resized.append(cv2.resize(cover, (120, 120)))
        
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
    
    #Calculate horizontal swipe movement for scrolling
    if previous_x is None:
        previous_x = x2 
        return

    delta_x = x2 - previous_x  # difference of current x with x-axis of previous frame
    previous_x = x2  #update previous_x with current frame

    # Detect swipe gesture
    if abs(delta_x) > 50:  # Swipe threshold = 50, most optimal for group's hand sizes and smooth gesture
        if delta_x > 0: # Swipe Right
            print("Right Swipe")
        else:  # Swipe Left
	        print("Left Swipe")

def update_shift():
    global shift_x
    if shift_x != swipe_target:
        step = swipe_speed if swipe_target > shift_x else -swipe_speed
        shift_x += step
        if abs(shift_x - swipe_target) < swipe_speed:
            shift_x = swipe_target

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

    # Generate dynamic positions for album covers
    positions = generate_positions(len(albums))

    # Overlay album covers
    for i, cover in enumerate(cover_images_resized):
        x, y = positions[i]["pos"]
        w, h = positions[i]["size"]
        cover_resized = cv2.resize(cover, (w, h))

        if x + w > 0 and x < screen_width:  # Draw only visible albums
            # Ensure the dimensions are valid and clip to the frame bounds
            frame_h, frame_w, _ = frame.shape
            x_end = min(x + w, frame_w)
            y_end = min(y + h, frame_h)
            x_start = max(x, 0)
            y_start = max(y, 0)

            # Calculate the region of the cover that fits within the visible frame
            cover_start_x = max(0, -x)
            cover_start_y = max(0, -y)
            cover_end_x = cover_start_x + (x_end - x_start)
            cover_end_y = cover_start_y + (y_end - y_start)

            # Only overlay if dimensions are valid
            if x_start < x_end and y_start < y_end:
                frame[y_start:y_end, x_start:x_end] = cover_resized[cover_start_y:cover_end_y, cover_start_x:cover_end_x]

    cv2.imshow('A5', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

