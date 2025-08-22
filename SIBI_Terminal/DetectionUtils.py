import os
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

STATIC_DATA_PATH = os.path.join('SIBI_Static_Data')

def mediapipe_detection(image, model):
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    # Make prediction
    results = model.process(image)
    # Convert back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    image_height, image_width, _ = image.shape

    # Draw and label left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
        
        # Get bounding box coordinates
        x_coords = [landmark.x for landmark in results.left_hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in results.left_hand_landmarks.landmark]
        if x_coords and y_coords: # Pastikan list tidak kosong
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Konversi ke koordinat piksel dan pastikan dalam batas gambar
            x_min_px = max(0, int(x_min * image_width) - 10) # Padding
            y_min_px = max(0, int(y_min * image_height) - 10) # Padding
            x_max_px = min(image_width, int(x_max * image_width) + 10) # Padding
            y_max_px = min(image_height, int(y_max * image_height) + 10) # Padding

            # Draw rectangle and label
            cv2.rectangle(image, (x_min_px, y_min_px), (x_max_px, y_max_px), (121, 22, 76), 2)
            cv2.putText(image, 'Kiri', (x_min_px, y_min_px - 5 if y_min_px > 10 else y_min_px + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (121, 22, 76), 2)

    # Draw and label right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        
        x_coords = [landmark.x for landmark in results.right_hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in results.right_hand_landmarks.landmark]
        if x_coords and y_coords: # Pastikan list tidak kosong
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            x_min_px = max(0, int(x_min * image_width) - 10)
            y_min_px = max(0, int(y_min * image_height) - 10)
            x_max_px = min(image_width, int(x_max * image_width) + 10)
            y_max_px = min(image_height, int(y_max * image_height) + 10)

            # Draw rectangle and label
            cv2.rectangle(image, (x_min_px, y_min_px), (x_max_px, y_max_px), (245, 117, 66), 2)
            cv2.putText(image, 'Kanan', (x_min_px, y_min_px - 5 if y_min_px > 10 else y_min_px + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 117, 66), 2)

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

def get_available_actions(data_path=STATIC_DATA_PATH):
    actions = []
    if os.path.exists(data_path):
        actions = [folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))]
    return np.array(actions) if actions else np.array([])