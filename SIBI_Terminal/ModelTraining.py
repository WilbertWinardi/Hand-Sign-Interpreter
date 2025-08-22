import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from DetectionUtils import get_available_actions, STATIC_DATA_PATH 

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def load_data(data_path=STATIC_DATA_PATH, frames_per_static_gesture=1):
    actions = get_available_actions(data_path)
    
    if len(actions) == 0:
        print(f"Tidak ada data ditemukan di {data_path}. Harap kumpulkan data terlebih dahulu.")
        return None, None, None, None, None
    
    print(f"Memuat data untuk actions: {actions}")
    
    label_map = {label:num for num, label in enumerate(actions)}
    
    features, labels = [], []
    
    for action in actions:
        action_dir = os.path.join(data_path, action)
        if not os.path.isdir(action_dir):
            continue
        # Loop melalui setiap contoh (example_folder adalah nomor contoh)
        for example_folder in os.listdir(action_dir):
            example_path = os.path.join(action_dir, example_folder)
            if os.path.isdir(example_path):
                window = [] # Ini akan menjadi sequence kita, panjangnya frames_per_static_gesture
                # Loop melalui setiap frame dalam contoh gestur statis ini
                for frame_num in range(frames_per_static_gesture):
                    frame_file_path = os.path.join(example_path, f"{frame_num}.npy")
                    if os.path.exists(frame_file_path):
                        res = np.load(frame_file_path)
                        window.append(res)
                    else:
                        print(f"Peringatan: File frame {frame_file_path} tidak ditemukan. Contoh ini mungkin diabaikan.")
                        pass 

                # Hanya tambahkan jika kita memiliki jumlah frame yang diharapkan
                if len(window) == frames_per_static_gesture:
                    features.append(window)
                    labels.append(label_map[action])
    
    if not features:
        print("Tidak ada fitur yang berhasil dimuat. Periksa struktur data Anda.")
        return None, None, None, None, None

    X = np.array(features)
    y = to_categorical(labels, num_classes=len(actions)).astype(int)
    
    if X.shape[0] == 0:
        print("Dataset kosong setelah pemrosesan.")
        return None, None, None, None, None

    print(f"Shape X sebelum split: {X.shape}") # (num_samples, frames_per_static_gesture, num_features_per_frame)
    print(f"Shape y sebelum split: {y.shape}") # (num_samples, num_classes)

    # Pastikan ada cukup sampel untuk split
    min_samples_for_split = 2 
    if X.shape[0] < min_samples_for_split or (len(np.unique(np.argmax(y, axis=1))) < 2 and X.shape[0] < 2 * len(actions)) : # Cek jika stratify mungkin gagal
        print("Tidak cukup sampel atau kelas untuk melakukan train-test split yang valid. Menggunakan semua data untuk training.")
        return X, None, y, None, actions # Kembalikan actions juga

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Data berhasil dimuat: {len(X_train)} train, {len(X_test)} test.")
    return X_train, X_test, y_train, y_test, actions

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False, activation='relu')) # False karena layer Dense berikutnya
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['categorical_accuracy'])
    return model

def train_model_main(X_train, y_train, X_test, y_test, actions_list, epochs=100, batch_size=32):
    if X_train is None or y_train is None or X_train.shape[0] == 0:
        print("Tidak ada data training yang tersedia atau data kosong.")
        return None
        
    log_dir = os.path.join('Logs_Static') # Folder log terpisah
    os.makedirs(log_dir, exist_ok=True)
    tb_callback = TensorBoard(log_dir=log_dir)
    # Early stopping untuk mencegah overfitting
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # input_shape adalah (frames_per_static_gesture, num_features_per_frame)
    model = build_model((X_train.shape[1], X_train.shape[2]), len(actions_list))
    
    model.summary()
    
    print(f"Memulai training model untuk {len(actions_list)} kelas: {actions_list}...")
    if X_test is not None and y_test is not None and X_test.shape[0] > 0:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test, y_test), 
                            callbacks=[tb_callback, early_stopping_callback])
        print("\nEvaluasi model pada data test...")
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {acc*100:.2f}%")
        print(f"Test Loss: {loss:.4f}")

        y_pred_probs = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred_probs, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_true_labels, y_pred_labels, target_names=actions_list, zero_division=0))

    else: # Jika tidak ada data test (misal sampel terlalu sedikit)
        print("Tidak ada data test, melatih hanya pada data training.")
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            callbacks=[tb_callback, early_stopping_callback])

    model_path = 'sibi_static_model.h5' # Nama model berbeda
    model.save(model_path)
    print(f"Model disimpan ke {model_path}")
    
    return model

if __name__ == '__main__':
    # Tentukan berapa frame yang membentuk satu gestur statis (harus sama dengan frames_per_capture)
    FRAMES_PER_GESTURE = 1 # Ubah ini jika Anda menggunakan >1 frame per gestur statis saat pengumpulan
    
    X_train, X_test, y_train, y_test, actions = load_data(data_path=STATIC_DATA_PATH, 
                                                          frames_per_static_gesture=FRAMES_PER_GESTURE)

    if X_train is not None and X_train.shape[0] > 0:
        trained_model = train_model_main(X_train, y_train, X_test, y_test, actions, epochs=150, batch_size=16)
        if trained_model:
            print("\nTraining model selesai.")
    else:
        print("\nTidak dapat melatih model karena masalah pemuatan data atau dataset kosong.")