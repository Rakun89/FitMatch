from django.urls import path
from . import views

urlpatterns = [
    # 1. Halaman User (Root)
    path('', views.user_index, name='user_index'),

    # 2. Halaman Dashboard Admin (YANG BARU)
    path('admin-panel/', views.admin_dashboard, name='admin_dashboard'),

    # 3. Halaman Training
    path('admin-training/', views.admin_training, name='admin_training'),
    path('run-training/', views.admin_run_training, name='admin_run_training'),

    # 4. Halaman Simulasi
    path('admin-simulation/', views.admin_simulation, name='admin_simulation'),
    
    # 5. Halaman Detail Simulasi (PENTING: Harus ada <int:id>)
    # Dashboard error biasanya karena baris ini belum ada atau namanya beda
    path('admin-simulation/<int:id>/', views.admin_simulation_detail, name='admin_simulation_detail'),
]