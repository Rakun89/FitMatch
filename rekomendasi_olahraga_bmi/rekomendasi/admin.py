from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import DatasetFile, TrainingLog, PredictionLog, DetailLatihan

# Daftarkan model agar muncul di http://127.0.0.1:8000/admin/
admin.site.register(DatasetFile)
admin.site.register(TrainingLog)
admin.site.register(PredictionLog)
admin.site.register(DetailLatihan)
class DetailLatihanAdmin(admin.ModelAdmin): # <--- Perhatikan bagian ini sudah diperbaiki
    list_display = ('nama_kategori', 'saran_durasi')
    search_fields = ('nama_kategori',)
