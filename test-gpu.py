import torch

# 1. Perintah utama untuk memeriksa ketersediaan CUDA
is_available = torch.cuda.is_available()

print(f"Apakah CUDA tersedia? {is_available}")

if is_available:
    # 2. Menampilkan jumlah GPU yang terdeteksi
    gpu_count = torch.cuda.device_count()
    print(f"Jumlah GPU yang terdeteksi: {gpu_count}")

    # 3. Menampilkan nama GPU yang sedang aktif (GPU 0)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Nama GPU: {gpu_name}")

    # 4. Menampilkan versi CUDA yang digunakan oleh PyTorch
    cuda_version = torch.version.cuda
    print(f"Versi CUDA yang digunakan PyTorch: {cuda_version}")
else:
    print("❌ Gagal mendeteksi GPU. PyTorch akan berjalan menggunakan CPU.")