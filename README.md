
# Aplikasi Deteksi Bahasa Isyarat SIBI (Statis)

Proyek ini adalah aplikasi untuk mendeteksi gestur statis dari Sistem Isyarat Bahasa Indonesia (SIBI) secara real-time menggunakan webcam. Aplikasi ini dibangun dengan Python dan memanfaatkan machine learning untuk mengenali berbagai gestur huruf.

Proyek ini memiliki dua mode utama:

1.  **Aplikasi Web**: Antarmuka yang ramah pengguna untuk melakukan deteksi SIBI secara langsung melalui browser.
2.  **Skrip Manajemen Lokal**: Sebuah antarmuka baris perintah (CLI) untuk developer guna mengumpulkan data gestur baru dan melatih model.

## Instalasi

Ikuti langkah-langkah berikut untuk menyiapkan lingkungan proyek.

1.  **Unduh Proyek** unduh dan ekstrak file proyek ke dalam satu folder.
        
2.  **Instal Dependensi** Instal semua library Python yang dibutuhkan dengan perintah berikut:
    
    Bash
    
    ```
    pip install tensorflow numpy opencv-python mediapipe flask flask-socketio eventlet Pillow scikit-learn
    
    ```
    

## Cara Menjalankan Aplikasi

Terdapat dua cara untuk menjalankan aplikasi ini, tergantung pada kebutuhan Anda.

----------

### **1. Menjalankan Aplikasi Web (Untuk Deteksi Real-time)**

Mode ini digunakan untuk pengguna akhir yang ingin mencoba deteksi bahasa isyarat.

**Sebelum Menjalankan:**

-   Pastikan file model `asl_static_model.h5` sudah ada di folder utama.
-   Pastikan folder `ASL_Static_Data` beserta subfolder data gestur (`a`, `b`, dst.) berada di lokasi yang benar (di dalam folder utama proyek). Ini penting agar aplikasi tahu label apa saja yang harus diprediksi.

**Langkah-langkah:**

1.  Buka terminal atau command prompt di direktori utama proyek.
2.  Jalankan server Flask dengan perintah:
    
    Bash
    
    ```
    python app.py
    
    ```
    
3.  Buka browser web Anda (misalnya Chrome atau Firefox).
4.  Kunjungi alamat `http://127.0.0.1:5000`.
5.  Izinkan browser untuk mengakses kamera Anda saat diminta.
6.  Aplikasi siap digunakan. Arahkan tangan Anda ke kamera untuk melihat prediksi.

----------

### **2. Menjalankan Skrip Lokal (Untuk Manajemen Data & Training)**

Mode ini digunakan oleh developer untuk mengelola dataset dan melatih model machine learning.

**Langkah-langkah:**

1.  Buka terminal atau command prompt di direktori utama proyek.
2.  Jalankan skrip `main.py` dengan perintah:
    
    Bash
    
    ```
    python main.py
    
    ```
    
3.  Sebuah menu akan tampil di terminal, memberikan Anda beberapa pilihan seperti:
    -   `1. Kumpulkan Data Training untuk Gestur Default (A-Z)`
    -   `2. Tambah Gestur Baru (Statis)`
    -   `3. Tambah Data ke Gestur yang Ada (Statis)`
    -   `4. Latih Model (Statis)`
    -   `5. Deteksi Real-time (Statis)` (Versi lokal menggunakan `cv2.imshow`)
    -   `6. Tampilkan Gestur yang Tersedia`
    -   `7. Keluar`
4.  Masukkan nomor pilihan Anda dan tekan Enter untuk melanjutkan. Ikuti instruksi yang muncul di layar untuk setiap menu.

## Penyelesaian Masalah (Troubleshooting)

-   **Prediksi Tidak Muncul di Web**: Jika aplikasi web berjalan tetapi tidak ada prediksi yang tampil, pastikan **pencahayaan** di ruangan Anda sangat baik dan tangan Anda memiliki **kontras yang jelas** dengan latar belakang. Ini adalah penyebab paling umum kegagalan deteksi tangan oleh MediaPipe.
-   **Error `IndexError` atau Aplikasi Crash**: Ini kemungkinan besar disebabkan oleh folder `ASL_Static_Data` yang kosong atau tidak ditemukan. Pastikan folder tersebut ada dan berisi data training Anda.

## Lampiran AOL
Link video demo : [https://binusianorgmy.sharepoint.com/personal/wilbert_winardi_binus_ac_id/_layouts/15/guestaccess.aspx?share=EYeqDE_viUpHmTE6XKYAJXAB_uqg85XYPH8gWViNuEYIEQ&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=rOngiD](https://binusianorg-my.sharepoint.com/personal/wilbert_winardi_binus_ac_id/_layouts/15/guestaccess.aspx?share=EYeqDE_viUpHmTE6XKYAJXAB_uqg85XYPH8gWViNuEYIEQ&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=rOngiD)

  

Link bukti wawancara : [https://binusianorgmy.sharepoint.com/personal/wilbert_winardi_binus_ac_id/_layouts/15/guestaccess.aspx?share=Ejo9FXESSW5Fq6Tv1oUaC9sBKxlaU0sE97KSmR4A0QYlLA&e=QchPK5](https://binusianorg-my.sharepoint.com/personal/wilbert_winardi_binus_ac_id/_layouts/15/guestaccess.aspx?share=Ejo9FXESSW5Fq6Tv1oUaC9sBKxlaU0sE97KSmR4A0QYlLA&e=QchPK5)

  
Link PPT Presentasi : [https://binusianorgmy.sharepoint.com/personal/wilbert_winardi_binus_ac_id/_layouts/15/guestaccess.aspx?share=EYawP_TAL1hJqZ6EOtzL5ywBqu3kkLcfkrwNP1R-sAlt9Q&e=SvKawq](https://binusianorg-my.sharepoint.com/personal/wilbert_winardi_binus_ac_id/_layouts/15/guestaccess.aspx?share=EYawP_TAL1hJqZ6EOtzL5ywBqu3kkLcfkrwNP1R-sAlt9Q&e=SvKawq)
"# Hand-Sign-Interpreter" 
