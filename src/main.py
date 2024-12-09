import pygame
pygame.mixer.init()
import cv2
import os
import time
import mediapipe as mp
import math as math
import tkinter as tk

# Suppress TensorFlow and MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

#for detecting horizontal swipe
previous_x = None  

#STATES
state = "NORMAL" 
# state= "EXPANDED"
# state= "PLAYING"

currentMiddleIndex = 0 # Start with the first album

#album box settings
smallDisplay = (100, 100)
largeDisplay = (300, 300)

swipe_target = 0  #target shift value for smooth transitions
swipe_speed = 20  #speed of the scrolling transition
shift_x = 0 

pinch_confirm_counter = 0
pinch_cooldown = 0.5

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
	    
def play_music(album, song_index):
    """Play the selected song."""
    pygame.mixer.music.stop()
    song_path = album["songs"][song_index]
    if os.path.exists(song_path):
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play()
        print(f"Playing: {song_path}")
    else:
        print(f"Error: Song file not found: {song_path}")

def stop_music():
    """Stop the currently playing music."""
    pygame.mixer.music.stop()
	
def draw_playlist(frame, album, selected_song_index):
    x, y = screen_width // 2 - 400, screen_height // 2 - 200
    cv2.putText(frame, f"{album['name']} - {album['artist']}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # Draw songs
    for i, song in enumerate(album["songs"]):
        color = (0, 255, 0) if i == selected_song_index else (255, 255, 255)
        cv2.putText(frame, song, (x, y + 50 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
	
def handle_gesture_controls(hand_landmarks):
    """Handle gestures """
    global selected_song_index, previous_x, swipe_target, currentMiddleIndex
    global last_pinch_time, state, last_played_song_index, pinch_confirm_counter

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Convert to pixel positions
    x1, y1 = int(thumb_tip.x * screen_width), int(thumb_tip.y * screen_height)
    x2, y2 = int(index_finger_tip.x * screen_width), int(index_finger_tip.y * screen_height)

    #Distance btw thumb and index finger
    pinch_distance = math.hypot(x2 - x1, y2 - y1)

    current_time = time.time()

     #If Pinch is detected
    if pinch_distance < 35:  # Pinch threshold
         
        pinch_confirm_counter += 1
        if pinch_confirm_counter >= 5: 
            if current_time - last_pinch_time > pinch_cooldown:  # Ensure cooldown
                if state == "PLAYING":
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.stop()
                        last_played_song_index = selected_song_index 
                        state = "EXPANDED"
                        print("Music stopped. Returned to EXPANDED state.")
                    else:
                        state = "NORMAL"
                        print("Exited to album list. State changed to NORMAL.")

                elif state == "EXPANDED":
                    if selected_song_index == last_played_song_index:
                        #if the same song is selected, exit to the album list
                        state = "NORMAL"
                        print("Exited to album list. State changed to NORMAL.")
                    else:
                        state = "PLAYING"
                        last_played_song_index = selected_song_index 
                        play_music(albums[currentMiddleIndex], selected_song_index)
                        print(f"Playing: {albums[currentMiddleIndex]['songs'][selected_song_index]}")

                elif state == "NORMAL":
                    state = "EXPANDED"
                    selected_song_index = 0  #default to the first song
                    print("Entered EXPANDED state.")
                
                #update the last pinch time
                last_pinch_time = current_time  
                pinch_confirm_counter = 0
            return  
    else:
        pinch_confirm_counter = 0
    # Handle swipe gestures (only in NORMAL state)
    if state == "NORMAL" and pinch_distance > 50:  # Ensure no pinch
        if previous_x is None:
            previous_x = x2 
            return

        delta_x = x2 - previous_x  #difference of current x with x-axis of previous frame
        previous_x = x2  #update previous_x with current frame
        # Detect swipe gesture
        if abs(delta_x) > 70:  # Swipe threshold
            if delta_x > 0:  #swipe right
                if currentMiddleIndex > 0:
                    currentMiddleIndex -= 1
                    swipe_target += screen_width // len(albums)  # Move right
                    print(f"Swiped right. Current album index: {currentMiddleIndex}")
            else:  #swipe left
                if currentMiddleIndex < len(albums) - 1:
                    currentMiddleIndex += 1
                    swipe_target -= screen_width // len(albums)  # Move left
                    print(f"Swiped left. Current album index: {currentMiddleIndex}")

    # Handle vertical movement for playlist in EXPANDED state
    if state == "EXPANDED" and pinch_distance > 50:  #to ensure no pinch
        song_y_position = int(index_finger_tip.y * screen_height)

        #scaling factor for slower vertical swiping
        vertical_scaling_factor = 2  #higher value makes swiping slower
        adjusted_y_position = song_y_position // vertical_scaling_factor

       
        playlist_start_y = screen_height // 2 - 200  #starting y-position of the playlist
        song_height = 40  #Approximatiately pixel height of each song in the list

        song_index = (adjusted_y_position - (playlist_start_y // vertical_scaling_factor)) // (song_height // vertical_scaling_factor)

       
        if 0 <= song_index < len(albums[currentMiddleIndex]["songs"]):
            if selected_song_index != song_index:  
                selected_song_index = song_index
                print(f"Selected song: {albums[currentMiddleIndex]['songs'][selected_song_index]}")

def update_shift():
    global shift_x
    if shift_x != swipe_target:
        step = swipe_speed if swipe_target > shift_x else -swipe_speed
        shift_x += step
        if abs(shift_x - swipe_target) < swipe_speed:
            shift_x = swipe_target

# MAIN
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

    if state == "NORMAL":
        
        box_width = screen_width // 8  #space between album positions
        center_x = screen_width // 4 #center of the screen

        for i, cover in enumerate(cover_images_resized):
            relative_position = i - currentMiddleIndex
            x = center_x + (relative_position * box_width) - (cover.shape[1] // 2)
            y = screen_height // 2 - cover.shape[0] // 2 

            if i == currentMiddleIndex:
                highlighted_size = 150
                cover_resized = cv2.resize(cover, (highlighted_size, highlighted_size))
                x = center_x - highlighted_size // 2 
                y = screen_height // 2 - highlighted_size // 2
                
                cv2.rectangle(frame, (x - 10, y - 10), (x + highlighted_size + 10, y + highlighted_size + 10), (0, 255, 0), 5)
            else:
                normal_size = 100
                cover_resized = cv2.resize(cover, (normal_size, normal_size))

            #draw only visible albums
            if x + cover_resized.shape[1] > 0 and x < screen_width:
                frame_h, frame_w, _ = frame.shape
                x_end = min(x + cover_resized.shape[1], frame_w)
                y_end = min(y + cover_resized.shape[0], frame_h)
                x_start = max(x, 0)
                y_start = max(y, 0)

                cover_start_x = max(0, -x)
                cover_start_y = max(0, -y)
                cover_end_x = cover_start_x + (x_end - x_start)
                cover_end_y = cover_start_y + (y_end - y_start)

                if x_start < x_end and y_start < y_end:
                    frame[y_start:y_end, x_start:x_end] = cover_resized[cover_start_y:cover_end_y, cover_start_x:cover_end_x]
        update_shift()

    elif state in ["EXPANDED", "PLAYING"]:
        target_size = 200 
        x = 30  
        y = screen_height // 2 - target_size // 2 - 200  
        
        cover = cover_images_resized[currentMiddleIndex]
        cover_h, cover_w, _ = cover.shape

        aspect_ratio = cover_w / cover_h
       
        if aspect_ratio > 1:  
            new_w = target_size
            new_h = int(target_size / aspect_ratio)
        else:  
            new_h = target_size
            new_w = int(target_size * aspect_ratio)

        cover_resized = cv2.resize(cover, (new_w, new_h))

        #calculate the position to place the resized cover
        x_start = x
        y_start = y + (target_size - new_h) // 2
        x_end = x_start + new_w
        y_end = y_start + new_h

        #checking for dimensions are valid 
        frame_h, frame_w, _ = frame.shape
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(frame_w, x_end)
        y_end = min(frame_h, y_end)

        #overlay the resized and centered cover onto the frame
        if x_start < x_end and y_start < y_end:
            frame[y_start:y_end, x_start:x_end] = cover_resized[:y_end - y_start, :x_end - x_start]

        # Draw the playlist on the right side
        draw_playlist(frame, albums[currentMiddleIndex], selected_song_index)
        
    cv2.imshow('A5', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
pygame.mixer.music.stop()
cv2.destroyAllWindows()

