import os
import sys
import io
import time
import cv2 
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for, stream_with_context, jsonify

# Import your project's modules
from DetectionUtils import get_available_actions, STATIC_DATA_PATH, mediapipe_detection, draw_landmarks, mp_holistic, extract_keypoints
from ModelTraining import build_model, load_data as load_static_data
from tensorflow.keras.callbacks import Callback
import tensorflow as tf

app = Flask(__name__)
os.makedirs(STATIC_DATA_PATH, exist_ok=True)
camera = cv2.VideoCapture(0)
holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- NEW: Global variables for Sentence Building Feature ---
current_sentence = []
latest_stable_prediction = ""

# ====================================================================
# Keras Callback (Unchanged)
# ====================================================================
class WebLoggerCallback(Callback):
    def _log_to_web(self, log_string):
        print(log_string, file=sys.__stdout__)
        sys.__stdout__.flush()
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_string = f"data: Epoch {epoch+1:03d}: loss: {logs.get('loss'):.4f} - acc: {logs.get('categorical_accuracy'):.4f}"
        if logs.get('val_loss'):
            log_string += f" -- val_loss: {logs.get('val_loss'):.4f} - val_acc: {logs.get('val_categorical_accuracy'):.4f}"
        self._log_to_web(log_string + "\n\n")
    def on_train_begin(self, logs=None): self._log_to_web("data: Model training has started...\n\n")
    def on_train_end(self, logs=None): self._log_to_web("data: Model training has finished.\n\n")

# ====================================================================
# Main and Data Collection Routes (Unchanged)
# ====================================================================
@app.route('/')
def index(): return render_template('index.html')

@app.route('/gestures')
def gestures():
    available_actions = get_available_actions(data_path=STATIC_DATA_PATH).tolist()
    return render_template('gestures.html', actions=available_actions)

@app.route('/collect', methods=['GET', 'POST'])
def collect():
    if request.method == 'POST':
        gesture_name = request.form.get('gesture_name', '').strip().lower()
        num_examples = int(request.form.get('num_examples', 30))
        if not gesture_name: return render_template('collect.html', error="Gesture name cannot be empty.")
        return redirect(url_for('collect_live', gesture_name=gesture_name, num_examples=num_examples))
    return render_template('collect.html')
    
@app.route('/collect_live/<string:gesture_name>/<int:num_examples>')
def collect_live(gesture_name, num_examples):
    start_num = get_next_example_number(STATIC_DATA_PATH, gesture_name)
    return render_template('collect_live.html', gesture_name=gesture_name, num_examples=num_examples, start_num=start_num)

@app.route('/video_feed_collect')
def video_feed_collect(): return Response(generate_simple_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_data/<string:gesture_name>/<int:example_num>')
def save_data(gesture_name, example_num):
    success, frame = camera.read()
    if not success: return {"status": "error", "message": "Could not read frame from camera."}
    image, results = mediapipe_detection(frame, holistic_model)
    keypoints = extract_keypoints(results)
    example_path = os.path.join(STATIC_DATA_PATH, gesture_name, str(example_num))
    os.makedirs(example_path, exist_ok=True)
    np.save(os.path.join(example_path, "0.npy"), keypoints)
    return {"status": "success", "message": f"Data saved for {gesture_name} - example {example_num}"}

# ====================================================================
# Model Training Route (Unchanged)
# ====================================================================
@app.route('/train')
def train(): return render_template('train.html')

@app.route('/run_training')
def run_training():
    # This remains the same as the previous working version
    def training_stream_generator():
        # ... (The full robust training logic from the previous step)
        try:
            yield "data: --- Loading Data ---\n\n"
            X_train, X_test, y_train, y_test, actions = load_static_data(data_path=STATIC_DATA_PATH)
            if X_train is None:
                yield "data: ERROR: No data found.\n\n"
                return
            yield f"data: --- Found {len(actions)} actions. Building model... ---\n\n"
            from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
            model = build_model((X_train.shape[1], X_train.shape[2]), len(actions))
            web_logger_callback = WebLoggerCallback()
            yield "data: Starting model.fit()...\n\n"
            model.fit(X_train, y_train, epochs=100, batch_size=2, validation_data=(X_test, y_test) if X_test is not None else None,
                      callbacks=[TensorBoard(log_dir=os.path.join('Logs_Static')), EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True), web_logger_callback],
                      verbose=0)
            model.save('sibi_static_model.h5')
            yield "data: \n--- Model saved ---\n\n"
        except Exception as e:
            import traceback
            yield f"data: An error occurred: {e}\n{traceback.format_exc()}\n\n"
    def stream_wrapper():
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            for log in training_stream_generator():
                output = sys.stdout.getvalue()
                if output:
                    yield output
                    sys.stdout.truncate(0)
                    sys.stdout.seek(0)
                yield log
        finally:
            sys.stdout = original_stdout
    return Response(stream_with_context(stream_wrapper()), mimetype='text/event-stream')


# ====================================================================
# Real-Time Detection Routes (UPDATED)
# ====================================================================
@app.route('/detect')
def detect():
    # Reset sentence when navigating to the page
    global current_sentence
    current_sentence = []
    return render_template('detect.html')

@app.route('/video_feed_detect')
def video_feed_detect():
    return Response(generate_detection_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- NEW ROUTES FOR SENTENCE BUILDING ---
@app.route('/add_letter', methods=['POST'])
def add_letter():
    global current_sentence, latest_stable_prediction
    if latest_stable_prediction:
        current_sentence.append(latest_stable_prediction)
    return jsonify({"sentence": "".join(current_sentence)})

@app.route('/delete_letter', methods=['POST'])
def delete_letter():
    global current_sentence
    if current_sentence:
        current_sentence.pop()
    return jsonify({"sentence": "".join(current_sentence)})

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global current_sentence
    current_sentence = []
    return jsonify({"sentence": ""})

# ====================================================================
# Helper Frame Generators (UPDATED)
# ====================================================================
def generate_simple_frames():
    # This is for collection and is unchanged
    while True:
        success, frame = camera.read()
        if not success: break
        image, _ = mediapipe_detection(frame, holistic_model)
        draw_landmarks(image, _)
        (flag, encodedImage) = cv2.imencode(".jpg", image)
        if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

# In app.py

def generate_detection_frames():
    """ Reverted to the original, every-frame prediction for responsiveness. """
    global latest_stable_prediction
    model_path = 'sibi_static_model.h5'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return
    
    model = tf.keras.models.load_model(model_path)
    actions = get_available_actions(STATIC_DATA_PATH)
    sequence_buffer, prediction_history = [], []
    
    while True:
        success, frame = camera.read()
        if not success: break
            
        image, results = mediapipe_detection(frame, holistic_model)
        draw_landmarks(image, results)
        keypoints = extract_keypoints(results)
        sequence_buffer.append(keypoints)
        sequence_buffer = sequence_buffer[-1:]

        text_to_draw = "Predicted: ..."
        latest_stable_prediction = "" # Reset on each frame

        if len(sequence_buffer) == 1:
            try:
                # --- THIS IS THE MISSING LINE THAT HAS BEEN ADDED BACK ---
                input_data = np.expand_dims(sequence_buffer, axis=0)
                
                prediction_probs = model.predict(input_data, verbose=0)[0]
                predicted_idx = np.argmax(prediction_probs)
                confidence = prediction_probs[predicted_idx]
                
                if confidence > 0.90: # High confidence threshold for stability
                    prediction = actions[predicted_idx]
                    latest_stable_prediction = prediction # Update global state
                    # NEW TEXT FORMAT
                    text_to_draw = f"Predicted: {prediction.upper()} ({confidence:.2f})"

            except Exception as e:
                print(f"Error during prediction: {e}")
        
        # Draw the text on the frame
        cv2.putText(image, text_to_draw, (15, image.shape[0] - 20), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        (flag, encodedImage) = cv2.imencode(".jpg", image)
        if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
# ====================================================================
# Main execution point
# ====================================================================
if __name__ == '__main__':
    # Need to get the correct get_next_example_number from DataCollection
    from DataCollection import get_next_example_number
    app.run(debug=True)