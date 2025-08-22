import os
import sys
import io
import time
import cv2 
from flask import Flask, render_template, Response, request, redirect, url_for, stream_with_context

# Import your project's modules
# Add mediapipe_detection, draw_landmarks, and mp_holistic for our helper function
from DetectionUtils import get_available_actions, STATIC_DATA_PATH, mediapipe_detection, draw_landmarks, mp_holistic
from DataCollection import collect_static_gesture_data, get_next_example_number
from ModelTraining import load_data as load_static_data, train_model_main
from SIBI_Detection import generate_frames

DEFAULT_FRAMES_PER_STATIC_GESTURE = 1 

def main_menu():
    print("\n" + "="*50)
    print("SISTEM DETEKSI SIBI (STATIS)")
    print("="*50)
    print("1. Kumpulkan Data Training untuk Gestur Default (A-Z)")
    print("2. Tambah Gestur Baru (Statis)")
    print("3. Tambah Data ke Gestur yang Ada (Statis)")
    print("4. Latih Model (Statis)")
    print("5. Deteksi Real-time (Statis)")
    print("6. Tampilkan Gestur yang Tersedia")
    print("7. Keluar")
    print("="*50)
    choice = input("Masukkan pilihan Anda (1-7): ")
    return choice

def collect_default_static_signs():
    default_actions = np.array(list(string.ascii_lowercase)) # a-z
    print(f"Anda akan mengumpulkan data untuk gestur default: {', '.join(default_actions)}")
    
    try:
        num_examples_per_action = int(input(f"Jumlah contoh per gestur (default: 20): ") or "20")
        frames_to_capture = int(input(f"Jumlah frame per contoh gestur (default: {DEFAULT_FRAMES_PER_STATIC_GESTURE}): ") or str(DEFAULT_FRAMES_PER_STATIC_GESTURE))
    except ValueError:
        num_examples_per_action = 20
        frames_to_capture = DEFAULT_FRAMES_PER_STATIC_GESTURE
        print(f"Input tidak valid. Menggunakan default: {num_examples_per_action} contoh, {frames_to_capture} frame per contoh.")

    if not os.path.exists(STATIC_DATA_PATH):
        os.makedirs(STATIC_DATA_PATH)
        print(f"Folder data dibuat: {STATIC_DATA_PATH}")

    for action_idx, action_name in enumerate(default_actions):
        print(f"\n--- Mengumpulkan untuk gestur: {action_name.upper()} ({action_idx+1}/{len(default_actions)}) ---")
        input(f"Tekan Enter untuk mulai mengumpulkan '{action_name.upper()}' atau 'q' lalu enter untuk skip gestur ini...")
        if input == 'q':
            print(f"Melewati gestur '{action_name.upper()}'")
            continue
            
        start_num = get_next_example_number(action_name, data_path=STATIC_DATA_PATH)
        collect_static_gesture_data(action_name, 
                                    num_examples=num_examples_per_action, 
                                    start_example_num=start_num,
                                    frames_per_capture=frames_to_capture,
                                    data_path=STATIC_DATA_PATH)
    print("\nPengumpulan data untuk gestur default selesai.")

def add_new_static_sign():
    sign_name = input("Masukkan nama gestur statis yang ingin ditambahkan (misal: 'a', 'b', 'ok'): ").strip().lower()
    if not sign_name:
        print("Nama gestur tidak boleh kosong.")
        return

    available_signs = get_available_actions(data_path=STATIC_DATA_PATH)
    if sign_name in available_signs:
        print(f"Gestur '{sign_name}' sudah ada.")
        choice = input("Apakah Anda ingin menambahkan lebih banyak data ke gestur ini? (y/n): ").strip().lower()
        if choice == 'y':
            add_more_data_to_static_sign(sign_name)
        return
    
    try:
        num_examples = int(input(f"Jumlah contoh untuk gestur '{sign_name}' (default: 30): ") or "30")
        frames_to_capture = int(input(f"Jumlah frame per contoh (default: {DEFAULT_FRAMES_PER_STATIC_GESTURE}): ") or str(DEFAULT_FRAMES_PER_STATIC_GESTURE))

    except ValueError:
        num_examples = 30
        frames_to_capture = DEFAULT_FRAMES_PER_STATIC_GESTURE
        print(f"Input tidak valid. Menggunakan default: {num_examples} contoh, {frames_to_capture} frame per contoh.")
        
    print(f"Anda akan merekam {num_examples} contoh untuk gestur '{sign_name}'.")
    input("Tekan Enter untuk memulai...")
    
    collect_static_gesture_data(sign_name, 
                                num_examples=num_examples, 
                                start_example_num=0, # Selalu mulai dari 0 untuk gestur baru
                                frames_per_capture=frames_to_capture,
                                data_path=STATIC_DATA_PATH)
    print(f"Pengumpulan data untuk gestur '{sign_name}' selesai.")

def add_more_data_to_static_sign(sign=None):
    available_signs = get_available_actions(data_path=STATIC_DATA_PATH)
    
    if len(available_signs) == 0:
        print("Tidak ada gestur tersedia. Harap kumpulkan data terlebih dahulu.")
        return
    
    if sign is None:
        print("Gestur yang tersedia:")
        for i, s_name in enumerate(available_signs):
            print(f"{i+1}. {s_name}")
        try:
            sign_idx = int(input("Pilih gestur (nomor): ")) - 1
            if not (0 <= sign_idx < len(available_signs)):
                print("Pilihan tidak valid.")
                return
            sign = available_signs[sign_idx]
        except ValueError:
            print("Input tidak valid.")
            return
    
    next_example = get_next_example_number(sign, data_path=STATIC_DATA_PATH)
    
    try:
        num_additional_examples = int(input(f"Jumlah contoh tambahan untuk '{sign}' (default: 10): ") or "10")
        frames_to_capture = int(input(f"Jumlah frame per contoh (default: {DEFAULT_FRAMES_PER_STATIC_GESTURE}, harus konsisten): ") or str(DEFAULT_FRAMES_PER_STATIC_GESTURE))
    except ValueError:
        num_additional_examples = 10
        frames_to_capture = DEFAULT_FRAMES_PER_STATIC_GESTURE
        print(f"Input tidak valid. Menggunakan default: {num_additional_examples} contoh, {frames_to_capture} frame.")
        
    print(f"Anda akan merekam {num_additional_examples} contoh tambahan untuk gestur '{sign}'.")
    print(f"Mulai dari contoh nomor {next_example}")
    input("Tekan Enter untuk memulai...")
    
    collect_static_gesture_data(sign, 
                                num_examples=num_additional_examples, 
                                start_example_num=next_example,
                                frames_per_capture=frames_to_capture,
                                data_path=STATIC_DATA_PATH)
    print(f"Pengumpulan data tambahan untuk gestur '{sign}' selesai.")

def show_available_static_signs():
    actions = get_available_actions(data_path=STATIC_DATA_PATH)
    if len(actions) == 0:
        print("Tidak ada gestur statis tersedia dalam sistem.")
    else:
        print("\nGestur statis yang tersedia:")
        for i, action in enumerate(actions):
            print(f"{i+1}. {action}")
        print(f"\nTotal: {len(actions)} gestur")
    input("\nTekan Enter untuk melanjutkan...")

def main():
    os.makedirs(STATIC_DATA_PATH, exist_ok=True) # Pastikan folder data statis ada
    
    while True:
        choice = main_menu()
        
        if choice == '1':
            collect_default_static_signs()
        elif choice == '2':
            add_new_static_sign()
        elif choice == '3':
            add_more_data_to_static_sign()
        elif choice == '4':
            print("Memuat data statis untuk training...")
            # frames_per_static_gesture harus sama dengan yang digunakan saat pengumpulan data
            X_train, X_test, y_train, y_test, actions_list = load_static_data(
                data_path=STATIC_DATA_PATH, 
                frames_per_static_gesture=DEFAULT_FRAMES_PER_STATIC_GESTURE
            )
            if X_train is not None and actions_list is not None and len(actions_list) > 0:
                train_model_main(X_train, y_train, X_test, y_test, actions_list, epochs=150, batch_size=16)
            else:
                print("Tidak dapat melatih model, data tidak tersedia atau kosong.")
            input("Tekan Enter untuk melanjutkan...")
        elif choice == '5':
            real_time_detection(model_path='sibi_static_model.h5', 
                                frames_for_prediction=DEFAULT_FRAMES_PER_STATIC_GESTURE,
                                data_path=STATIC_DATA_PATH)
        elif choice == '6':
            show_available_static_signs()
        elif choice == '7':
            print("Keluar dari program...")
            break
        else:
            print("Pilihan tidak valid. Silakan coba lagi.")

if __name__ == "__main__":
    main()