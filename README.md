# hci_assignment5
HCI Assignment - 5

In this project, a hand gesture-controlled PC-based music player which allows users to navigate through albums, select songs, and control playback with simple hand gestures, is implemented using Mediapipe. For this purpose, the main objectives are to design a user-friendly music player interface that allows users to control a music player without any physical contact and to ensure that hand movements are perceived very sensitively and accurately. 

# Implementation Details

Tools and Libraries that are used are listed below:
Python
MediaPipe
OpenCV
Pygame

For programming languages, Python is chosen for implementation due to simplicity and for having an easy integration of different tools. For image and video processing, OpenCV is imported. MediaPipe is used for tracking and detecting the gestures as stated in the documentation of Assignment-5. Since a hand gesture-controlled music player is implemented, Pygame is used for handling music playback.

The implemented interface has the ability to detect 3 different gestures. First one is horizontal scrolling, which helps to navigate through albums inside the interface. Second gesture is vertical scrolling and it is used to go through songs of an album. The last gesture, pinching, enables the user to pick albums and open/close specific songs.

To implement the music tool properly, three different states are applied. First state is “NORMAL” which displays the album list. This state is like a default state. When the user opens the interface for the first time, s/he sees the “NORMAL” state at the beginning. The second one is “EXPANDED” which displays the selected album cover on the left and its song playlist on the right. The last state is “PLAYING” which keeps the album view while the music is playing. 

On the other hand, to control the music, 3 functionalities are implemented: play, pause and stop. Play functionality starts a selected song automatically, pause functionality stops playback and retains the current song position, and stop functionality ends playback and exits the song list.

For interface design, albums are displayed as a list which is horizontally scrollable. The album can be selected by a pinch gesture. When a user selects the album, the album cover and songs playlist are displayed on the screen. The album cover is displayed on the left, and the playlist is displayed on the right. The user can choose the songs with a vertical scrolling hand gesture and select music to play by the pinch gesture. 

The implementation begins with the capturing video feed with the help of OpenCV. Thumb and index finger and similar hand gestures are detected and tracked by MediaPipe Hands. It enables the detection of pinching and vertical and horizontal scrolling gestures. According to the recognized gesture, the system switches between the states to manage “NORMAL”, “EXPANDED” and “PLAYING”. In “NORMAL” state, the album list is displayed and users can select albums to be selected in the album list using horizontal swipe gestures. All music is not displayed in a list because it will increase the memory load. To decrease the memory load, the chunking method is used for music so that the music is grouped in albums which is a very effective way of reducing memory load (Benyon, 2020). When there is a pinch gesture by the user, the system goes to “EXPANDED” state and shows the selected album cover on the left side and the song list on the right side. Users can navigate between songs with vertical scrolls. If the user is in the song list and pinches on a song, it plays the selected song. Another pinch gesture pauses the selected playing song. Play and pause operations are performed by Pygame. If the user wants to go to another album, if they stay on the same song without changing the song and perform a pinch gesture, the song list is exited and the album list is returned.

