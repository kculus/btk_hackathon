@echo off
echo 🐇 OPTİMİZE QLoRA EĞİTİMİ BAŞLATILIYOR...
echo ================================================
echo ⚡ Terminal önerileri kullanılarak optimize edildi
echo 📊 2 epoch, eager attention, mixed precision
echo ================================================

echo.
echo 📦 Gerekli paketler kontrol ediliyor...
pip install -r requirements_qlora.txt

echo.
echo 🚀 Optimize eğitim başlatılıyor...
python train_qlora_gemma_optimized.py

echo.
echo ✅ Eğitim tamamlandı!
pause 