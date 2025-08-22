import os
import cv2
import numpy as np
import time 
from DetectionUtils import mediapipe_detection, draw_landmarks, extract_keypoints, mp_holistic, STATIC_DATA_PATH

def get_next_example_number(action, data_path=STATIC_DATA_PATH):
    action_path = os.path.join(data_path, action)
    
    if not os.path.exists(action_path):
        return 0 
    
    existing_examples = []
    for folder_name in os.listdir(action_path):
        if os.path.isdir(os.path.join(action_path, folder_name)):
            try:
                existing_examples.append(int(folder_name))
            except ValueError:
                pass # Abaikan folder yang namanya bukan angka
                
    return max(existing_examples) + 1 if existing_examples else 0

def collect_static_gesture_data(action, num_examples=30, start_example_num=None, 
                                countdown_duration=0, # Default countdown dihilangkan
                                frames_per_capture=1, data_path=STATIC_DATA_PATH):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera.")
        return

    if start_example_num is None:
        current_example_num = get_next_example_number(action, data_path)
    else:
        current_example_num = start_example_num
        
    end_example_num = current_example_num + num_examples

    action_main_path = os.path.join(data_path, action)
    try:
        os.makedirs(action_main_path, exist_ok=True)
    except Exception as e:
        print(f"Error membuat direktori {action_main_path}: {e}")
        cap.release()
        cv2.destroyAllWindows()
        return

    # Inisialisasi counter untuk contoh yang benar-benar dikumpulkan dalam sesi ini
    successfully_collected_this_session = 0

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while current_example_num < end_example_num:
            print(f"\nSiap untuk gestur: {action.upper()} | Contoh: {current_example_num} dari {start_example_num + num_examples -1}")
            
            if countdown_duration > 0:
                for countdown in range(countdown_duration, 0, -1):
                    ret, frame_countdown = cap.read()
                    if not ret or frame_countdown.size == 0:
                        print("Gagal mengambil frame dari kamera saat countdown.")
                        time.sleep(0.5)
                        continue
                    
                    display_frame_countdown = frame_countdown.copy()
                    cv2.putText(display_frame_countdown, f'Siap-siap untuk: {action.upper()}', (int(display_frame_countdown.shape[1]*0.1), int(display_frame_countdown.shape[0]*0.4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.putText(display_frame_countdown, f'Contoh {current_example_num} mulai dalam {countdown}...', (int(display_frame_countdown.shape[1]*0.1), int(display_frame_countdown.shape[0]*0.5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.imshow('Pengumpulan Data Statis', display_frame_countdown)
                    if cv2.waitKey(1000) & 0xFF == ord('q'): 
                        cap.release()
                        cv2.destroyAllWindows()
                        print("Pengumpulan data dihentikan oleh pengguna.")
                        return
            
            captured_this_example = False
            while not captured_this_example:
                ret, frame = cap.read()
                if not ret or frame.size == 0:
                    print("Gagal mengambil frame dari kamera.")
                    time.sleep(0.1) 
                    continue
                
                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)
                
                cv2.putText(image, f'GESTUR: {action.upper()}', (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'CONTOH: {current_example_num}', (15, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"TEKAN 'S' utk SIMPAN, 'Q' utk KELUAR", (15, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow('Pengumpulan Data Statis', image)
                
                key = cv2.waitKey(10) & 0xFF
                if key == ord('s'): 
                    print(f"Menyimpan contoh {current_example_num} untuk gestur '{action}'...")
                    example_path = os.path.join(data_path, action, str(current_example_num))
                    try:
                        os.makedirs(example_path, exist_ok=True)
                    except Exception as e:
                        print(f"Error membuat direktori {example_path}: {e}")
                        continue

                    frame_buffer_for_capture = frame 
                    results_buffer_for_capture = results

                    saved_frames_count = 0
                    for frame_idx in range(frames_per_capture):
                        current_frame_to_save = frame_buffer_for_capture
                        current_results_for_save = results_buffer_for_capture

                        if frame_idx > 0: 
                            ret_new, frame_new = cap.read()
                            if not ret_new or frame_new.size == 0:
                                print(f"Gagal mengambil frame tambahan {frame_idx} yang valid untuk contoh {current_example_num}")
                                continue # Lewati frame ini, atau `break` jika ini kritis
                            current_frame_to_save = frame_new
                            _, current_results_for_save = mediapipe_detection(frame_new, holistic)
                        
                        keypoints = extract_keypoints(current_results_for_save)
                        npy_path = os.path.join(example_path, str(frame_idx))
                        np.save(npy_path, keypoints)
                        # Hapus print detail frame agar konsol tidak terlalu ramai saat spam 's'
                        # print(f"  -> Frame {frame_idx} disimpan ke {npy_path}.npy") 
                        saved_frames_count +=1

                        if frames_per_capture > 1: # Umpan balik visual sangat singkat jika >1 frame
                            display_capture_frame = current_frame_to_save.copy()
                            draw_landmarks(display_capture_frame, current_results_for_save)
                            text_y_pos = display_capture_frame.shape[0] - 40 if display_capture_frame.shape[0] > 150 else 150
                            cv2.putText(display_capture_frame, f'Menyimpan frame {frame_idx+1}/{frames_per_capture}', 
                                        (15, text_y_pos),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1,cv2.LINE_AA)
                            cv2.imshow('Pengumpulan Data Statis', display_capture_frame)
                            cv2.waitKey(1) 
                    
                    if saved_frames_count == frames_per_capture:
                        print(f"Contoh {current_example_num} untuk '{action}' berhasil disimpan.")
                        successfully_collected_this_session += 1
                    else:
                        print(f"Peringatan: Contoh {current_example_num} untuk '{action}' mungkin tidak lengkap ({saved_frames_count}/{frames_per_capture} frame).")

                    captured_this_example = True
                    current_example_num += 1

                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Pengumpulan data dihentikan oleh pengguna.")
                    print(f"Total contoh berhasil dikumpulkan sesi ini untuk '{action}': {successfully_collected_this_session}.")
                    return
            
            if current_example_num >= end_example_num:
                break
                
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nPengumpulan data selesai untuk gestur '{action}'.")
    print(f"Total contoh berhasil dikumpulkan sesi ini untuk '{action}': {successfully_collected_this_session}.")