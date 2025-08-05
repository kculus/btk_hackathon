# 🐇 Matematik QLoRA Fine-tuning

Bu proje, Gemma-2-9b-it-tr-new modelini matematik konuları verileriyle QLoRA (Quantized Low-Rank Adaptation) kullanarak fine-tuning eder.

## 📋 Gereksinimler

- Python 3.8+
- CUDA uyumlu GPU (en az 8GB VRAM önerilir)
- 16GB+ RAM

## 🚀 Kurulum

1. **Gerekli kütüphaneleri yükle:**
```bash
pip install -r requirements_qlora.txt
```

2. **Eğitimi başlat:**
```bash
# Windows için
start_qlora_training.bat

# Veya manuel olarak
python train_qlora_gemma.py
```

## 📁 Dosya Yapısı

```
├── train_qlora_gemma.py          # Ana eğitim dosyası
├── test_qlora_model.py           # Model test dosyası
├── requirements_qlora.txt         # Gerekli kütüphaneler
├── start_qlora_training.bat      # Windows başlatma dosyası
├── gemma-2-9b-it-tr-new/        # Base model
├── gemma-matematik-qlora/        # Fine-tuned model (eğitim sonrası)
└── mat*.json                     # Matematik konuları verileri
```

## ⚙️ Konfigürasyon

### QLoRA Ayarları
- **Rank (r):** 16
- **Alpha:** 32
- **Dropout:** 0.05
- **Target Modules:** q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj

### Eğitim Ayarları
- **Epochs:** 3
- **Batch Size:** 2
- **Gradient Accumulation:** 4
- **Learning Rate:** 2e-4
- **Warmup Steps:** 100

### Quantization
- **4-bit quantization** (BitsAndBytes)
- **Double quantization** aktif
- **NF4** quantization type
- **bfloat16** compute dtype

## 📊 Veri Formatı

Eğitim verileri şu formatta olmalıdır:
```json
[
  {
    "instruction": "Soru veya talimat",
    "output": "Beklenen cevap"
  }
]
```

## 🧪 Model Testi

Eğitim tamamlandıktan sonra modeli test etmek için:

```bash
python test_qlora_model.py
```

## 💾 Model Kaydetme

Fine-tuned model `gemma-matematik-qlora/` dizinine kaydedilir:
- `adapter_config.json` - LoRA konfigürasyonu
- `adapter_model.safetensors` - LoRA ağırlıkları
- `tokenizer.json` - Tokenizer dosyaları

## 🔧 Kullanım

### Eğitim Sonrası Model Yükleme

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Base model yükle
base_model = AutoModelForCausalLM.from_pretrained("./gemma-2-9b-it-tr-new")
tokenizer = AutoTokenizer.from_pretrained("./gemma-2-9b-it-tr-new")

# LoRA adapter'ını yükle
model = PeftModel.from_pretrained(base_model, "./gemma-matematik-qlora")
```

### Cevap Üretme

```python
prompt = "<start_of_turn>user\nMatematik sorusu<end_of_turn>\n<start_of_turn>model\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 📈 Performans İpuçları

1. **GPU Memory:** Eğitim sırasında GPU memory kullanımını izleyin
2. **Gradient Checkpointing:** Büyük modeller için aktif
3. **Mixed Precision:** fp16 kullanarak hızlandırma
4. **Batch Size:** GPU memory'ye göre ayarlayın

## 🐛 Sorun Giderme

### Yaygın Hatalar

1. **CUDA Out of Memory:**
   - Batch size'ı düşürün
   - Gradient accumulation steps'i artırın
   - Model'i CPU'ya yükleyin

2. **Import Errors:**
   - `pip install -r requirements_qlora.txt` çalıştırın
   - CUDA versiyonunu kontrol edin

3. **Training Loss NaN:**
   - Learning rate'i düşürün
   - Gradient clipping ekleyin

## 📝 Notlar

- Eğitim süresi GPU'ya bağlı olarak 2-6 saat sürebilir
- Model boyutu yaklaşık 18GB (base model) + ~100MB (LoRA weights)
- Eğitim sonrası model matematik konularında daha iyi performans gösterecek

## 🎯 Sonuç

Bu fine-tuning işlemi sonrasında model:
- ✅ Matematik konularını daha iyi anlayacak
- ✅ Türkçe matematik terimlerini daha iyi kullanacak
- ✅ Sınıf seviyesine uygun cevaplar üretecek
- ✅ Eğitim verilerindeki formatı takip edecek 