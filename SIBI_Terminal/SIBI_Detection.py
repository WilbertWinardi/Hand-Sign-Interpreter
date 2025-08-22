import cv2
import numpy as np
import tensorflow as tf
import os
from DetectionUtils import mediapipe_detection, draw_landmarks, extract_keypoints, mp_holistic, get_available_actions, STATIC_DATA_PATH

def real_time_detection(model_path='sibi_static_model.h5', frames_for_prediction=1, data_path=STATIC_DATA_PATH):
    actions = get_available_actions(data_path)
    
    if len(actions) == 0:
        print(f"Tidak ada gestur yang dilatih dari {data_path}. Harap kumpulkan data dan latih model terlebih dahulu.")
        return
    
    if not os.path.exists(model_path):
        print(f"File model {model_path} tidak ditemukan. Harap latih model terlebih dahulu.")
        return
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model '{model_path}' berhasil dimuat. Gestur yang tersedia: {actions}")
    except Exception as e:
        print(f"Error memuat model: {e}")
        return
    
    # Pengaturan untuk pengumpulan frame dan prediksi
    sequence_buffer = [] # Buffer untuk frames_for_prediction
    displayed_sentence = [] # Kalimat yang dibentuk pengguna
    
    # Untuk stabilisasi prediksi sebelum ditambahkan ke kalimat
    prediction_history = [] 
    STABILIZATION_THRESHOLD = 0.7 # Minimal confidence
    STABILIZATION_WINDOW = 3 # Jumlah prediksi terakhir untuk dipertimbangkan

    current_stable_prediction = "" 
    current_confidence = 0.0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("Error: Tidak bisa membuka kamera.")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Gagal mengambil frame.")
                break
            
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results) # Ini akan menggambar tangan Kiri/Kanan + kotak
            
            keypoints = extract_keypoints(results)
            sequence_buffer.append(keypoints)
            sequence_buffer = sequence_buffer[-frames_for_prediction:]
            
            if len(sequence_buffer) == frames_for_prediction:
                # Input untuk model adalah (1, frames_for_prediction, num_features)
                input_data = np.expand_dims(sequence_buffer, axis=0)
                prediction_probs = model.predict(input_data)[0]
                
                predicted_idx = np.argmax(prediction_probs)
                confidence = prediction_probs[predicted_idx]
                
                if confidence > 0.5: # Filter awal berdasarkan confidence
                    prediction_history.append(actions[predicted_idx])
                    prediction_history = prediction_history[-STABILIZATION_WINDOW:]

                    # Cek apakah prediksi stabil
                    if len(prediction_history) == STABILIZATION_WINDOW:
                        most_common_prediction = max(set(prediction_history), key=prediction_history.count)
                        # Cek apakah prediksi paling umum muncul mayoritas dalam window
                        if prediction_history.count(most_common_prediction) >= STABILIZATION_WINDOW // 2 + 1 :
                             # Perbarui prediksi stabil jika berbeda atau confidence lebih tinggi
                            if most_common_prediction != current_stable_prediction or confidence > current_confidence:
                                current_stable_prediction = most_common_prediction
                                # Ambil confidence dari prediksi terbaru yang cocok dengan most_common
                                # Atau rata-ratakan confidence jika mau (lebih kompleks)
                                current_confidence = confidence # Gunakan confidence terbaru untuk kesederhanaan

                else: # Jika confidence rendah, reset prediksi stabil
                    current_stable_prediction = ""
                    current_confidence = 0.0

            if current_stable_prediction and current_confidence >= STABILIZATION_THRESHOLD:
                pred_text = f"Prediksi: {current_stable_prediction.upper()} ({current_confidence:.2f})"
                cv2.putText(image, pred_text, (15, image.shape[0] - 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                 cv2.putText(image, "Prediksi: ...", (15, image.shape[0] - 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

            sentence_display_text = "".join(displayed_sentence) # Gabungkan huruf tanpa spasi
            cv2.rectangle(image, (0,0), (image.shape[1], 40), (245, 117, 16), -1)
            cv2.putText(image, sentence_display_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, "SPACE: Tambah | BACKSPACE: Hapus | ESC: Hapus Semua | Q: Keluar", 
                        (15, image.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow('Deteksi SIBI Statis Real-Time', image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '): # Spacebar
                if current_stable_prediction and current_confidence >= STABILIZATION_THRESHOLD:
                    displayed_sentence.append(current_stable_prediction.upper()) # Tambah huruf yang stabil
                    current_stable_prediction = "" # Reset untuk prediksi berikutnya
                    current_confidence = 0.0
                    prediction_history.clear() # Bersihkan history agar tidak langsung tambah lagi
                    print(f"Ditambahkan: {displayed_sentence[-1]}. Kalimat: {''.join(displayed_sentence)}")
            elif key == 8: # Backspace
                if displayed_sentence:
                    removed = displayed_sentence.pop()
                    print(f"Dihapus: {removed}. Kalimat: {''.join(displayed_sentence)}")
            elif key == 27: # Escape (untuk Clear All)
                displayed_sentence.clear()
                print("Kalimat dibersihkan.")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    real_time_detection(model_path='sibi_static_model.h5', frames_for_prediction=1)