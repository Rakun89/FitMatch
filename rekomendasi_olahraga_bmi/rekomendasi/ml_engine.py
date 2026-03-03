import os
import joblib
import pandas as pd
import numpy as np
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class BrainAI:
    def __init__(self):
        # 1. Menentukan lokasi penyimpanan file model
        self.model_path = os.path.join(settings.BASE_DIR, 'ml_model/model_knn_bmi.pkl')

    # ==========================================
    # BAGIAN 1: PERSIAPAN DATA (PRE-PROCESSING)
    # ==========================================
    
    def _siapkan_kamus_data(self, df):
        """Membuat kamus (mapping) untuk mengubah Huruf -> Angka"""
        # Ambil daftar unik nama olahraga dari data
        daftar_olahraga = sorted(df['Exercise Routine'].unique())
        
        # Buat mapping
        maps = {
            'gender': {'Male': 1, 'Female': 0, 'Pria': 1, 'Wanita': 0},
            'bmi': {
                'Underweight': 1, 'Normal Weight': 2, 'Overweight': 3,
                'Obese Class 1': 4, 'Obese Class 2': 5, 'Obese Class 3': 6
            },
            'activity': {'Sedentary': 1, 'Lightly Active': 2, 'Moderately Active': 3, 'Active': 4},
            
            # Mapping Target (Olahraga) dibuat otomatis sesuai isi data
            'exercise': {nama: i+1 for i, nama in enumerate(daftar_olahraga)}
        }
        
        # Buat kamus kebalikannya (Angka -> Huruf) untuk keperluan prediksi nanti
        maps['exercise_rev'] = {angka: nama for nama, angka in maps['exercise'].items()}
        
        return maps

    def _bersihkan_data(self, df):
        """Membuang data yang tidak masuk akal (Cleaning)"""
        
        # Cek 1: Rumus Fisika (Berat/Tinggi^2 harus mendekati angka BMI di kolom)
        rumus_fisika = abs(df['BMI'] - (df['Weight'] / df['Height']**2)) < 0.1
        df = df[rumus_fisika]

        # Cek 2: Logika Label (Apakah angka BMI cocok dengan Kategori-nya?)
        def cek_label_sesuai(row):
            b, label = row['BMI'], row['BmiClass']
            if b < 18.5 and label == 'Underweight': return True
            if 18.5 <= b < 25 and label == 'Normal Weight': return True
            if 25 <= b < 30 and label == 'Overweight': return True
            if 30 <= b < 35 and label == 'Obese Class 1': return True
            if 35 <= b < 40 and label == 'Obese Class 2': return True
            if b >= 40 and label == 'Obese Class 3': return True
            return False
        
        # Hanya simpan data yang lolos pengecekan
        df_bersih = df[df.apply(cek_label_sesuai, axis=1)].copy()
        return df_bersih

    def _ubah_ke_angka(self, df, maps):
        """Mengubah (Encode) data teks menjadi angka agar bisa dihitung komputer"""
        df['Gender_Enc'] = df['Gender'].map(maps['gender'])
        df['BmiClass_Enc'] = df['BmiClass'].map(maps['bmi'])
        df['Activity_Enc'] = df['Activity Level'].map(maps['activity'])
        df['Exercise_Enc'] = df['Exercise Routine'].map(maps['exercise'])
        
        # Hapus baris kosong jika ada yang gagal diubah
        return df.dropna()

    # ==========================================
    # BAGIAN 2: PROSES TRAINING (INTI AI)
    # ==========================================

    def train_model(self, csv_path):
        """Fungsi Utama: Mengatur alur dari baca data sampai simpan model"""
        try:
            # LANGKAH 1: BACA & BERSIHKAN DATA
            df_mentah = pd.read_csv(csv_path)
            maps = self._siapkan_kamus_data(df_mentah)     # Siapkan kamus
            df_bersih = self._bersihkan_data(df_mentah)    # Buang data error
            df_final = self._ubah_ke_angka(df_bersih, maps) # Ubah huruf ke angka

            # LANGKAH 2: MEMBAGI DATA (Ujian & Latihan)
            # X = Soal (Umur, BMI, Gender, Aktivitas)
            # y = Jawaban (Olahraga)
            X = df_final[['Age', 'BmiClass_Enc', 'Gender_Enc', 'Activity_Enc']]
            y = df_final['Exercise_Enc']
            
            # 80% untuk latihan, 20% untuk ujian (test)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # LANGKAH 3: MENCARI K TERBAIK (LOOPING)
            # Kita coba K=1 sampai K=20, mana yang nilainya paling bagus?
            laporan_grafik = {'acc': [], 'err': [], 'prec': [], 'rec': []}
            best_k = 5
            best_score = 0
            
            for k in range(1, 21):
                # Coba training dengan k tetangga
                model_sementara = KNeighborsClassifier(n_neighbors=k)
                model_sementara.fit(X_train, y_train)
                
                # Uji model
                acc_test = model_sementara.score(X_test, y_test) * 100
                y_pred_loop = model_sementara.predict(X_test)
                
                # Simpan skor untuk grafik
                laporan_grafik['acc'].append(acc_test)
                laporan_grafik['err'].append(100 - acc_test)
                laporan_grafik['prec'].append(precision_score(y_test, y_pred_loop, average='weighted', zero_division=0)*100)
                laporan_grafik['rec'].append(recall_score(y_test, y_pred_loop, average='weighted', zero_division=0)*100)
                
                # Cek apakah ini rekor baru?
                if acc_test > best_score:
                    best_score = acc_test
                    best_k = k

            # LANGKAH 4: FINALISASI MODEL
            # Setelah ketemu K terbaik, kita buat model finalnya
            final_model = KNeighborsClassifier(n_neighbors=best_k)
            final_model.fit(X_train, y_train)
            
            # Hitung detail performa (Confusion Matrix & F1)
            y_pred_final = final_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred_final)
            f1 = f1_score(y_test, y_pred_final, average='weighted', zero_division=0) * 100
            
            # Siapkan nama kelas untuk label tabel
            nama_kelas = [maps['exercise_rev'][i] for i in final_model.classes_]

            # Kumpulkan semua metrics untuk dikirim ke Admin Dashboard
            metrics = {
                'accuracy': round(best_score, 2),
                'precision': round(laporan_grafik['prec'][best_k-1], 2), # Ambil dari list sesuai index K
                'recall': round(laporan_grafik['rec'][best_k-1], 2),
                'f1_score': round(f1, 2),
                'best_k': best_k,
                'graph_error': laporan_grafik['err'],
                'graph_acc_test': laporan_grafik['acc'],
                'graph_precision': laporan_grafik['prec'],
                'graph_recall': laporan_grafik['rec']
            }

            # LANGKAH 5: SIMPAN (SAVE)
            # Simpan model beserta kamus datanya agar bisa dipakai nanti
            payload = {
                'model': final_model, 
                'maps': maps,
                'X_train_data': X_train, 
                'y_train_data': y_train
            }
            joblib.dump(payload, self.model_path)

            return {
                'status': 'success', 
                'metrics': metrics,
                'confusion_matrix': cm.tolist(),
                'classes': nama_kelas,
                'total_data_test': len(y_test)
            }

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    # ==========================================
    # BAGIAN 3: PREDIKSI & FITUR ADMIN
    # ==========================================

    def predict_user(self, age, gender_code, bmi_cat, activity_code):
        """Dipakai User: Tebak olahraga berdasarkan input"""
        try:
            if not os.path.exists(self.model_path): return None
            data = joblib.load(self.model_path)
            
            # Ubah BMI Category (teks) jadi Angka
            bmi_code = data['maps']['bmi'].get(bmi_cat, 2)
            
            # Lakukan Prediksi
            input_data = [[age, bmi_code, int(gender_code), int(activity_code)]]
            pred_code = data['model'].predict(input_data)[0]
            
            # Kembalikan dalam bentuk Teks (Nama Olahraga)
            return data['maps']['exercise_rev'].get(pred_code, "Tidak Diketahui")
        except: return None

    def get_calculation_details(self, age, gender_code, bmi_cat, activity_code):
        """Dipakai Admin: Lihat siapa saja tetangga terdekatnya (White Box)"""
        try:
            if not os.path.exists(self.model_path): return []
            payload = joblib.load(self.model_path)
            model = payload['model']
            X_train = payload['X_train_data']
            y_train = payload['y_train_data']
            maps = payload['maps']
            
            # Siapkan input
            bmi_code = maps['bmi'].get(bmi_cat, 2)
            # Pastikan urutan input SAMA PERSIS dengan saat training: 
            # ['Age', 'BmiClass_Enc', 'Gender_Enc', 'Activity_Enc']
            input_data = [[age, bmi_code, int(gender_code), int(activity_code)]]
            
            # Cari tetangga
            distances, indices = model.kneighbors(input_data)
            
            # Susun laporan tetangga
            neighbors_list = []
            for i, idx in enumerate(indices[0]):
                row = X_train.iloc[idx]
                target = y_train.iloc[idx]
                
                # Kembalikan angka ke huruf agar mudah dibaca admin
                gender_lbl = "Pria" if row['Gender_Enc'] == 1 else "Wanita"
                target_lbl = maps['exercise_rev'].get(target, "-")
                
                # Cari nama BMI dari kodenya (Reverse lookup)
                bmi_lbl = [k for k, v in maps['maps']['bmi'].items() if v == row['BmiClass_Enc']][0] if 'maps' in maps else [k for k, v in maps['bmi'].items() if v == row['BmiClass_Enc']][0]

                # --- BAGIAN BARU: AMBIL LABEL ACTIVITY ---
                # Kita cari nama Activity (misal: 'Active') berdasarkan angkanya (misal: 4)
                act_lbl = [k for k, v in maps['activity'].items() if v == row['Activity_Enc']][0]
                # -----------------------------------------
                
                neighbors_list.append({
                    'rank': i + 1,
                    'jarak': round(distances[0][i], 4),
                    'age': int(row['Age']),
                    'gender': gender_lbl,
                    'bmi_category': bmi_lbl,
                    'activity': act_lbl,      # <--- JANGAN LUPA MASUKKAN KE SINI
                    'activity_code': int(row['Activity_Enc']), # Opsional: jika butuh angkanya
                    'class_hasil': target_lbl
                })
            return neighbors_list
        except Exception as e: 
            print(f"Error: {e}")
            return []