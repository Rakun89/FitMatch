from django.db import models

# Create your models here.
from django.db import models

# ========================
# 1. TABEL UNTUK ADMIN
# ========================
class DatasetFile(models.Model):
    """Menyimpan file CSV yang diupload Admin"""
    nama_file = models.CharField(max_length=100, default="Dataset Baru")
    file_csv = models.FileField(upload_to='datasets/') 
    tanggal_upload = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.nama_file} ({self.tanggal_upload.strftime('%d/%m/%Y')})"

class TrainingLog(models.Model):
    # ... field lama biarkan ...
    timestamp = models.DateTimeField(auto_now_add=True)
    jumlah_data = models.IntegerField()
    best_k = models.IntegerField()
    
    # METRICS
    akurasi = models.FloatField()          # Ini Testing Accuracy
    train_akurasi = models.FloatField(default=0) # <--- TAMBAHAN BARU
    
    precision = models.FloatField()
    recall = models.FloatField()
    status = models.CharField(max_length=50)

    def __str__(self):
        return f"Log {self.timestamp}"

class DetailLatihan(models.Model):
    # DAFTAR PILIHAN SATUAN (Single Exercise)
    JENIS_SATUAN_CHOICES = [
        ('Cardio', 'Cardio'),
        ('Strength Training', 'Strength Training'),
        ('Low Impact Cardio', 'Low Impact Cardio'),
        ('Water Aerobics', 'Water Aerobics'),
        ('Weight Gain Program', 'Weight Gain Program'),
        ('Medical Supervised Workouts', 'Medical Supervised Workouts'),
        ('Low Impact Workouts', 'Low Impact Workouts'),
    ]

    nama_kategori = models.CharField(
        max_length=100, 
        unique=True, 
        choices=JENIS_SATUAN_CHOICES, # <--- Pilihan Satuan
        verbose_name="Jenis Olahraga Satuan"
    )
    
    deskripsi = models.TextField(verbose_name="Penjelasan")
    contoh_gerakan = models.TextField(help_text="Pisahkan dengan Enter")
    saran_durasi = models.CharField(max_length=100, default="30-45 Menit")
    gambar = models.ImageField(upload_to='olahraga/', blank=True, null=True)

    def __str__(self):
        return self.nama_kategori

    def get_list_gerakan(self):
        return self.contoh_gerakan.split('\n')
# ========================
# 2. TABEL UNTUK USER
# ========================
class PredictionLog(models.Model):
    """Menyimpan Riwayat User Menggunakan Sistem"""
    timestamp = models.DateTimeField(auto_now_add=True)
    nama_user = models.CharField(max_length=100)
    gender = models.CharField(max_length=20)
    age = models.IntegerField()
    bmi_score = models.FloatField()
    bmi_category = models.CharField(max_length=50)
    hasil_prediksi = models.CharField(max_length=100)
    activity_level = models.IntegerField(default=1)

    def __str__(self):
        return f"{self.nama_user} - {self.hasil_prediksi}"