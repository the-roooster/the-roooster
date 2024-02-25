import cv2
import mediapipe as mp
import time
import math
import screen_brightness_control as sbc

# Set the width and height of the camera feed
wCam, hCam = 1280, 720

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initialize variables for time tracking
ctime = 0
ptime = 0

# Main loop for processing frames
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find hands
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks for thumb and index finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert landmarks to pixel coordinates
            thumb_x, thumb_y = int(thumb_tip.x * wCam), int(thumb_tip.y * hCam)
            index_x, index_y = int(index_tip.x * wCam), int(index_tip.y * hCam)

            # Calculate the distance between thumb and index finger
            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)

            # Adjust the volume based on the distance
            if distance < 50:
                sbc.set_brightness(0)
            elif distance < 100:
                sbc.set_brightness(25)
            elif distance < 150:
                sbc.set_brightness(50)
            elif distance < 200:
                sbc.set_brightness(75)
            elif distance < 250:
                sbc.set_brightness(100)
            else:
                sbc.set_brightness(70)

    # Calculate and display FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(frame, "FPS: " + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
