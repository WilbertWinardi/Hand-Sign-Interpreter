import cv2
import numpy as np
import tensorflow as tf
import os
from DetectionUtils import mediapipe_detection, draw_landmarks, extract_keypoints, mp_holistic, get_available_actions, STATIC_DATA_PATH

def generate_frames(model_path='sibi_static_model.h5'):
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return

    try:
        model = tf.keras.models.load_model(model_path)
        actions = get_available_actions(STATIC_DATA_PATH)
        print(f"Model '{model_path}' loaded successfully. Actions: {actions}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    sequence_buffer = []
    prediction_history = []
    displayed_sentence = []
    
    STABILIZATION_THRESHOLD = 0.7
    STABILIZATION_WINDOW = 3
    current_stable_prediction = ""

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)

            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence_buffer.append(keypoints)
            sequence_buffer = sequence_buffer[-1:] # Using 1 frame for static

            if len(sequence_buffer) == 1:
                input_data = np.expand_dims(sequence_buffer, axis=0)
                prediction_probs = model.predict(input_data)[0]
                predicted_idx = np.argmax(prediction_probs)
                confidence = prediction_probs[predicted_idx]

                if confidence > 0.5:
                    prediction_history.append(actions[predicted_idx])
                    prediction_history = prediction_history[-STABILIZATION_WINDOW:]

                    if len(prediction_history) == STABILIZATION_WINDOW:
                        most_common = max(set(prediction_history), key=prediction_history.count)
                        if prediction_history.count(most_common) >= STABILIZATION_WINDOW // 2 + 1:
                            if confidence > STABILIZATION_THRESHOLD:
                                current_stable_prediction = most_common

                if current_stable_prediction:
                     pred_text = f"Prediction: {current_stable_prediction.upper()} ({confidence:.2f})"
                     cv2.putText(image, pred_text, (15, image.shape[0] - 70), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", image)
            if not flag:
                continue

            # Yield the output frame in the byte format
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
    
    cap.release()