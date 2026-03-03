import json
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages 
from django.db.models import Count
from .models import DatasetFile, TrainingLog, PredictionLog, DetailLatihan
from .ml_engine import BrainAI


# A. Prepration 
MAPPING_OLAHRAGA = {
    'Mixed Cardio and Strength Training': ['Cardio', 'Strength Training'],
    'Low Impact Cardio, Strength Training': ['Low Impact Cardio', 'Strength Training'],
    'Water Aerobics, Low Impact Workouts': ['Water Aerobics', 'Low Impact Workouts'],
    'Strength Training, Weight Gain Program': ['Strength Training', 'Weight Gain Program'],
    'Cardio, Strength Training': ['Cardio', 'Strength Training'],
    'Medical Supervised Workouts': ['Medical Supervised Workouts'],
}

def _hitung_kategori_bmi(bmi_value):
    """Menentukan teks kategori berdasarkan angka BMI"""
    if bmi_value < 18.5: return "Underweight"
    elif bmi_value < 25: return "Normal Weight"
    elif bmi_value < 30: return "Overweight"
    elif bmi_value < 35: return "Obese Class 1"
    elif bmi_value < 40: return "Obese Class 2"
    return "Obese Class 3"

def _encode_gender(gender_text):
    """Mengubah teks gender menjadi angka (1: Pria, 0: Wanita)"""
    list_pria = ['1', '1.0', 'male', 'pria', 'laki-laki', 'man']
    return 1 if str(gender_text).lower().strip() in list_pria else 0

def _encode_bmi(bmi_category):
    """Mengubah kategori BMI menjadi angka skala 1-6"""
    mapping = {
        'Underweight': 1, 'Normal Weight': 2, 'Overweight': 3,
        'Obese Class 1': 4, 'Obese Class 2': 5, 'Obese Class 3': 6
    }
    return mapping.get(bmi_category, 2) # Default Normal

def _siapkan_data_elbow(metrics):
    """Menyiapkan data grafik Elbow untuk template"""
    acc_list = metrics.get('graph_acc_test', [])
    err_list = metrics.get('graph_error', [])
    best_k = metrics.get('best_k', 0)
    
    data = []
    for i, (acc, err) in enumerate(zip(acc_list, err_list)):
        k_val = i + 1
        data.append({
            'k': k_val,
            'accuracy': round(acc, 2),
            'error': round(err, 2),
            'is_best': (k_val == best_k)
        })
    return data

def _siapkan_detail_kelas(cm, classes, total_test):
    """Menghitung Precision & Recall per kelas dari Confusion Matrix"""
    details = []
    if not cm: return details, 0, 0

    total_correct = sum([cm[i][i] for i in range(len(cm))])
    total_wrong = total_test - total_correct
    total_samples = sum(sum(row) for row in cm)

    for idx, class_name in enumerate(classes):
        tp = cm[idx][idx]
        fp = sum([row[idx] for row in cm]) - tp
        fn = sum(cm[idx]) - tp
        tn = total_samples - (tp + fp + fn)
        
        prec = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        rec  = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        
        details.append({
            'name': class_name,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'precision': round(prec, 2),
            'recall': round(rec, 2)
        })
    return details, total_correct, total_wrong

# B. VIEW ADMIN

@login_required(login_url='/admin/login/')
def admin_dashboard(request):
    # Ambil Data Statistik
    last_model = TrainingLog.objects.filter(status="Berhasil").last()
    
    # Ambil Data Grafik BMI & Gender
    def get_chart_data(field):
        stats = PredictionLog.objects.values(field).annotate(total=Count(field))
        return [item[field] for item in stats], [item['total'] for item in stats]

    bmi_labels, bmi_data = get_chart_data('bmi_category')
    gender_labels, gender_data = get_chart_data('gender')

    context = {
        'total_prediksi': PredictionLog.objects.count(),
        'total_dataset': DatasetFile.objects.count(),
        'akurasi_model': last_model.akurasi if last_model else 0,
        'recent_logs': PredictionLog.objects.all().order_by('-timestamp')[:5],
        # JSON Dumps langsung di sini agar HTML bersih
        'bmi_labels': json.dumps(bmi_labels),
        'bmi_data': json.dumps(bmi_data),
        'gender_labels': json.dumps(gender_labels),
        'gender_data': json.dumps(gender_data),
    }
    return render(request, 'admin/dashboard.html', context)


@login_required(login_url='/admin/login/')
def admin_training(request):
    result = request.session.get('training_result', None)
    context = {'dataset_terakhir': DatasetFile.objects.last()}

    if result:
        # Gunakan Helper Function agar code rapi
        metrics = result.get('metrics', {})
        cm = result.get('confusion_matrix', [])
        classes = result.get('classes', [])
        
        total_data = metrics.get('data_count', 0)
        jumlah_uji = result.get('total_data_test', 0)
        if total_data == 0 and jumlah_uji > 0:
            total_data = int(jumlah_uji / 0.2)
        jumlah_latih = total_data - jumlah_uji
        
        elbow_data = _siapkan_data_elbow(metrics)
        class_details, correct, wrong = _siapkan_detail_kelas(cm, classes, jumlah_uji)

        context.update({
            'result': result,
            'jumlah_latih': jumlah_latih,
            'jumlah_uji': jumlah_uji,
            'total_data': total_data,
            'total_correct': correct,
            'total_wrong': wrong,
            'cm_json': json.dumps(cm),
            'classes_json': json.dumps(classes),
            'class_details': class_details,
            'elbow_data': elbow_data 
        })
    
    return render(request, 'admin/training.html', context)


@login_required(login_url='/admin/login/')
def admin_run_training(request):
    if request.method == 'POST':
        dataset = DatasetFile.objects.last()
        if dataset:
            brain = BrainAI()
            result = brain.train_model(dataset.file_csv.path)

            if result['status'] == 'success':
                m = result['metrics']
                TrainingLog.objects.create(
                    jumlah_data=m.get('data_count', 0),
                    best_k=m.get('best_k', 0),
                    akurasi=m.get('accuracy', 0),
                    train_akurasi=m.get('train_accuracy', 0),
                    precision=m.get('precision', 0),
                    recall=m.get('recall', 0), 
                    status="Berhasil"
                )
                request.session['training_result'] = result
                messages.success(request, f"Training Selesai! Akurasi: {m.get('accuracy',0)}%")
            else:
                messages.error(request, f"Error: {result.get('message', 'Unknown')}")
            
    return redirect('admin_training')



# C. VIEW SIMULASI (Detail Perhitungan / White Box)

@login_required(login_url='/admin/login/')
def admin_simulation(request):
    return render(request, 'admin/simulation.html', {
        'logs': PredictionLog.objects.all().order_by('-timestamp'),
        'model': TrainingLog.objects.filter(status="Berhasil").last()
    })

@login_required(login_url='/admin/login/')
def admin_simulation_detail(request, id):
    log = PredictionLog.objects.get(id=id)
    
    # --- 1. MAPPING AKTIVITAS ---
    map_act_text = {1: 'Sedentary', 2: 'Lightly Active', 3: 'Moderately Active', 4: 'Active'}
    act_code = getattr(log, 'activity', 2) 
    act_text = map_act_text.get(act_code, 'Lightly Active')

    # --- 2. SIAPKAN VECTOR USER (P) ---
    # Kita siapkan versi Text (untuk Label) dan Code (untuk Rumus/AI)
    user_vector = {
        'age': log.age, 
        'gender': log.gender,
        'gender_code': _encode_gender(log.gender), # Angka (1/0)
        'bmi': log.bmi_category,
        'bmi_code': _encode_bmi(log.bmi_category), # Angka (1-6)
        'activity': act_text,
        'activity_code': int(act_code)             # Angka (1-4)
    }

    # --- 3. AMBIL DATA TETANGGA (Q) ---
    brain = BrainAI()
    # Panggil dengan urutan argumen yang benar (Positional Arguments)
    neighbors = brain.get_calculation_details(
        user_vector['age'], 
        user_vector['gender_code'], 
        user_vector['bmi'], 
        user_vector['activity_code']
    ) 
    
    # --- 4. SIAPKAN VECTOR TETANGGA TERDEKAT ---
    neighbor_vector = {}
    if neighbors:
        n1 = neighbors[0]
        neighbor_vector = {
            'age': n1.get('age', 0),
            'gender': n1.get('gender', '-'),
            'gender_code': _encode_gender(n1.get('gender', '-')), # Convert balik ke angka
            'bmi': n1.get('bmi_category', '-'),
            'bmi_code': _encode_bmi(n1.get('bmi_category', '-')), # Convert balik ke angka
            'activity': n1.get('activity', '-'),
            'activity_code': n1.get('activity_code', 0),
            'jarak': n1.get('jarak', 0),
            'label': n1.get('class_hasil') or "-"
        }

    # --- 5. RENDER ---
    context = {
        'log': log,
        'neighbors': neighbors,
        'k_value': len(neighbors) if neighbors else 0,
        'user_vector': user_vector,         
        'neighbor_vector': neighbor_vector,
        'neighbors_json': json.dumps([{'x': n['age'], 'y': n['jarak']} for n in neighbors] if neighbors else []),
    }
    return render(request, 'admin/simulation_detail.html', context)



# D. VIEW USER (Frontend)

def user_index(request):
    context = {}
    
    if request.method == 'POST':
        try:
            # 1. Ambil Data
            nama = request.POST.get('nama')
            usia = int(request.POST.get('usia'))
            gender_code = request.POST.get('gender') # 1 atau 0
            berat = float(request.POST.get('berat'))
            tinggi = float(request.POST.get('tinggi')) / 100
            
            # Ambil Aktivitas (Pastikan form HTML mengirim value angka 1-4)
            aktivitas_code = int(request.POST.get('aktivitas', 2)) 

            # 2. Hitung BMI & Kategori
            bmi_score = berat / (tinggi ** 2)
            kategori_bmi = _hitung_kategori_bmi(bmi_score)

            # 3. Prediksi AI
            brain = BrainAI()
            # Kirim parameter aktivitas_code ke fungsi predict
            hasil_latihan = brain.predict_user(usia, gender_code, kategori_bmi, aktivitas_code) or "Hubungi Admin"

            # 4. Simpan Log (UPDATE BAGIAN INI)
            PredictionLog.objects.create(
                nama_user=nama, 
                age=usia, 
                gender="Pria" if str(gender_code) == '1' else "Wanita",
                bmi_score=round(bmi_score, 2), 
                bmi_category=kategori_bmi,
                activity_level=aktivitas_code,
                hasil_prediksi=hasil_latihan
            )

            # 5. Cari Konten Edukasi
            target_kategori = MAPPING_OLAHRAGA.get(hasil_latihan, [hasil_latihan])
            list_konten = DetailLatihan.objects.filter(nama_kategori__in=target_kategori)

            context = {
                'hasil': True, 'nama': nama, 
                'bmi': round(bmi_score, 2), 'kategori': kategori_bmi, 
                'latihan': hasil_latihan, 'list_detail_konten': list_konten
            }
        except Exception as e:
            context = {'error': f"Terjadi kesalahan: {str(e)}"}

    return render(request, 'user/index.html', context)