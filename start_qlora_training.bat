@echo off
echo 🐇 Matematik QLoRA Fine-tuning Başlatılıyor...
echo ================================================

echo 📦 Gerekli kütüphaneler yükleniyor...
pip install -r requirements_qlora.txt

echo 🚀 Eğitim başlatılıyor...
python train_qlora_gemma.py

echo ✅ Eğitim tamamlandı!
pause 