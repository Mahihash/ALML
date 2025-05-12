import numpy as np
import cv2
import mediapipe as mp
import pygame
import time
from datetime import datetime


pygame.mixer.init()
beep_sound = pygame.mixer.Sound("beep-01a.wav") 

# Pose setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()  # Exit if the camera isn't accessible

counter = 0
stage = None
last_beep_time = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break  # Break the loop if no frame is captured
        
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # RIGHT side
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # LEFT side
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angles
            angle_r = calculate_angle(r_shoulder, r_elbow, r_wrist)
            angle_l = calculate_angle(l_shoulder, l_elbow, l_wrist)

            # Both arms down
            if angle_r > 160 and angle_l > 160:
                stage = "down"
            # Both arms pressed up
            if angle_r < 70 and angle_l < 70 and stage == "down":
                stage = "up"
                counter += 1
                current_time = time.time()
                if current_time - last_beep_time > 1:
                    beep_sound.play()
                    last_beep_time = current_time
                print(f"Reps: {counter}")

            # Rep + Stage UI
            cv2.rectangle(image, (0, 0), (250, 100), (245, 117, 16), -1)
            cv2.putText(image, 'REPS', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(image, str(counter), (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

            cv2.putText(image, 'STAGE', (100, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(image, stage if stage else "", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        except Exception as e:
            print("Tracking error:", e)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Dual Shoulder Press Counter', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Save result to a text file
cap.release()
cv2.destroyAllWindows()
today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open("shoulder_press_log.txt", "a") as file:
    file.write(f"{today} - Total Shoulder Press Reps: {counter}\n")

print(f"Workout complete! Reps saved: {counter}")
