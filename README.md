# 🧠 Chatbot sistemi – `neuralwork/gemma-2-9b-it-tr` + QLoRA + RAG

Bu proje, **Hugging Face'teki `neuralwork/gemma-2-9b-it-tr`** modelini baz alır. Türkçe içeriklere özel LoRA tabanlı fine-tuning ve Retrieval‑Augmented Generation (RAG) ile belge destekli sohbet sistemi geliştirilmiştir.

---

## 🧪 Temel Model: `neuralwork/gemma-2-9b-it-tr`

- `google/gemma-2-9b-it` üzerine **yaklaşık 55k soru-cevap ve konuşma örneği ile fine-tune edilmiş** bir modeldir :contentReference[oaicite:1]{index=1}.
- LoRA parametreleri: `rank=128`, `lora_alpha=64`; eğitim süresi ~4 gün, RTX 6000 Ada GPU kullanılmıştır :contentReference[oaicite:2]{index=2}.
- Model, Türkçe'de daha iyi konuşma ve mantıksal çözümleme yeteneklerine sahiptir.

- Bu model LoRa ile fine tune edildi ve konu anlatımı yapabilen chatbot oluşturuldu. Bu aşamada kullanılan veri seti ve oluşan modelin linki aşağıda yer almaktadır.
- Model linki: https://huggingface.co/yagiz1323/EducationalChatbot/tree/main
- Model için kullanılan veri seti: https://huggingface.co/datasets/yagiz1323/FinetuningTurkishMat/tree/main

---
# 🧠 Advanced RAG Sistemi - Detaylı Açıklama

Bu proje, **Türkçe matematik eğitimi** için geliştirilmiş gelişmiş bir **Retrieval-Augmented Generation (RAG)** sistemi kullanmaktadır. Sistem, öğrencilerin matematik sorularını anlayıp doğru ve kapsamlı cevaplar üretebilmek için tasarlanmıştır.

##  Sistem Mimarisi

### 1. **Temel Bileşenler**

#### 📚 **Veri Tabanı (Knowledge Base)**
- **Matematik konuları**: 1-8. sınıf matematik müfredatı
- **Doküman formatı**: JSON dosyaları (`mat1.json`, `mat2.json`, ..., `mat8.json`)
- **İçerik türü**: Konu anlatımları, örnek sorular, çözümler
- **Toplam doküman sayısı**: 2,185 adet matematik içeriği

#### 🔍 **Embedding Model**
- **Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Boyut**: 384-dimensional vector space
- **Özellik**: Çok dilli (multilingual) destek
- **Kullanım**: Dokümanları ve sorguları vektör uzayına dönüştürme

#### 🤖 **LLM (Large Language Model)**
- **Model**: `gemma-2-9b-it-tr-new` (Türkçe fine-tune edilmiş)
- **Boyut**: 9B parametre
- **Optimizasyon**: RTX 4090 için özel memory yönetimi
- **Kullanım**: Cevap üretimi ve sorgu optimizasyonu

### 2. **RAG Pipeline Adımları**

####  **1. Inquiry (Sorgu Alma)**
```python
inquiry = query  # Kullanıcı sorgusu
```

####  **2. Pre-Process (Sorgu Optimizasyonu)**
```python
optimized_query = self.preprocess_query_with_llm(inquiry)
```
- LLM ile sorguyu matematik terimleri için optimize etme
- Anahtar kelimeleri çıkarma
- Sorgu netliğini artırma

####  **3. Embed (Vektörleştirme)**
```python
query_embedding = self.advanced_rag.create_embedding_for_text(optimized_query)
```
- Sorguyu 384-dimensional vektöre dönüştürme
- Semantic similarity için hazırlama

#### 🔍 **4. Search (Doküman Arama)**
```python
retrieved_docs = self.advanced_rag.retrieve_relevant_documents(optimized_query, top_k=5)
```
- FAISS index kullanarak en ilgili 3-5 dokümanı bulma
- Cosine similarity ile sıralama
- Embedding skoru hesaplama

####  **5. Filter (Doküman Filtreleme)**
```python
filtered_docs = self.filter_documents_with_llm(inquiry, retrieved_docs)
```
- LLM ile en uygun dokümanları seçme
- Kalite kontrolü
- İlgisiz içerikleri eleme

#### **6. Build Prompt (Prompt Oluşturma)**
```python
context_prompt = self.build_rag_prompt(inquiry, filtered_docs)
```
- Dokümanları birleştirme
- Çocukların anlayabileceği formatta prompt hazırlama
- Türkçe matematik terminolojisi kullanma

#### ⚡ **7. Generate (Cevap Üretimi)**
```python
initial_response = self.generate_with_llm(context_prompt, max_length=300, temperature=0.7)
```
- LLM ile cevap üretme
- Sıcaklık parametresi ile yaratıcılık kontrolü
- Token limiti ile uzunluk kontrolü

#### 🔄 **8. Self-Reflect (Kalite Kontrolü)**
```python
final_response = self.self_reflect_and_improve(inquiry, initial_response, filtered_docs)
```
- Cevap kalitesini değerlendirme
- Gerekirse iyileştirme
- Embedding skorları ile doğrulama

### 3. **Özel Özellikler**

#### 🎓 **Sınıf Bazlı Öğrenme**
```python
self.class_topics = {
    "1": ["📊 Sayılar (1-100)", "➕ Toplama İşlemi", ...],
    "2": ["�� Sayılar (1-1000)", "✖️ Çarpma İşlemi", ...],
    # ... 8. sınıfa kadar
}
```

#### 🧠 **Memory Management**
- **Kullanıcı mesajları**: Son 7 mesaj
- **Bot mesajları**: Son 7 cevap
- **Konuşma geçmişi**: Context için kullanım
- **Memory temizleme**: Her 3 mesajda bir GPU memory temizliği

#### ⚡ **GPU Optimizasyonları**
```python
# RTX 4090 için özel ayarlar
torch.cuda.set_per_process_memory_fraction(0.85)  # GPU'nun %85'i
torch.cuda.empty_cache()  # Memory temizleme
max_memory={0: "4GB"}  # Memory limiti
```

### 4. **Sistem Dosyaları**

#### **Core Files**
- **🧠 Eğitilmiş model ve index dosyaları Linki**: [AdvancedRagSystemFiles Dataset](https://mega.nz/folder/5Jw1xR6Y#ZjmVueMOEhEjlhJ86hB6xQ)
- `advanced_rag.py`: Ana RAG sistemi
- `advanced_rag_chat_fixed.py`: Chatbot implementasyonu
- `advanced_rag_system/`: Eğitilmiş model ve index dosyaları

#### **Data Files**
- **📊 Veri Seti Linki**: [AdvancedRagMatQuestions Dataset](https://huggingface.co/datasets/yagiz1323/AdvancedRagMatQuestions/tree/main)
- `mat1.json` - `mat8.json`: Sınıf bazlı matematik içerikleri
- `mat1konu.json` - `mat8konu.json`: Konu anlatımları
- `mat8_lgs.json`: LGS hazırlık içerikleri

#### **Web Interface**
- `app.py`: Flask web uygulaması
- `templates/`: HTML şablonları
- `static/`: CSS, JS, ses dosyaları

### 5. **Performans Özellikleri**

#### **Hız Optimizasyonları**
- **FAISS Index**: Hızlı doküman arama
- **GPU Acceleration**: CUDA ile hızlandırma
- **Memory Management**: Agresif memory yönetimi
- **Batch Processing**: Toplu işlem desteği

#### 🎯 **Kalite Kontrolü**
- **Embedding Skorları**: Semantic similarity kontrolü
- **Self-Reflection**: LLM ile kalite değerlendirmesi
- **Document Filtering**: İlgisiz içerik eleme
- **Response Validation**: Cevap doğrulama

### 6. **Kullanım Senaryoları**

#### **Konu Anlatımı**
```
Kullanıcı: "Kesirlerde toplama nasıl yapılır?"
Bot: [RAG sistemi ile kesir toplama konusunu açıklar]
```

#### 🧮 **Soru Çözümü**
```
Kullanıcı: "2/3 + 1/4 = ?"
Bot: [Adım adım çözümü gösterir]
```

#### 🎓 **Sınıf Seviyesi Belirleme**
```
Kullanıcı: "Ben 3. sınıfa gidiyorum"
Bot: [3. sınıf konularını listeler]
```

### 7. **Teknik Detaylar**

####  **Model Konfigürasyonu**
```python
# Embedding Model
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedding_dimension = 384

# LLM Model
model_path = "./gemma-2-9b-it-tr-new"
max_memory = "4GB"
temperature = 0.7
max_length = 300
```

#### 📊 **Veri İstatistikleri**
- **Toplam doküman**: 2,185 adet
- **Sınıf bazlı dağılım**: 1-8. sınıf
- **Embedding boyutu**: 384-dimensional
- **FAISS index boyutu**: 3.2MB
- **Model boyutu**: 449MB

Bu sistem, Türkçe matematik eğitimi için özel olarak tasarlanmış, gelişmiş bir RAG pipeline'ıdır. Öğrencilerin matematik sorularını anlayıp doğru ve kapsamlı cevaplar üretebilmek için optimize edilmiştir.
🎮 Oyunlaştırılmış Matematik Deneyimi
- Proje, sadece konu anlatımı ve sohbet ile sınırlı kalmaz; aynı zamanda etkileşimli bir matematik oyunu da sunar.

- Kullanıcıya ekranda sorular gelir, doğru yanıt verdikçe zorluk seviyesi artar.
- Cevaplar farklı lokasyonlarda yer alır; hız ve dikkat önemlidir.
- Doğru yanıtlarla puan kazanılır, zor sorular daha fazla puan kazandırır.
- Arka planda çalan müzikler ve ses efektleri, tamamen AI ile üretilmiştir.

Bu sayede öğrenciler, eğlenerek öğrenme deneyimi yaşar.

📖 Etkileşimli Hikaye Üretimi
- Projede yer alan AI destekli hikaye oluşturma modülü, kullanıcının yaratıcılığını teşvik eden interaktif bir yapıya sahiptir.
- Kullanıcı, kendi hikaye başlangıcını yazabilir veya sistemin sunduğu rastgele 2 başlangıçtan birini seçebilir.
- Hikaye, 10 adımdan oluşur. Her adımda kullanıcıya 2 farklı seçenek sunulur ve seçimlere göre hikaye yönlenir.
- Son adımda, yapay zekâ tarafından oluşturulmuş 2 alternatif son gösterilir.
- Tamamlanan hikaye:
- Sesli olarak dinlenebilir (TTS teknolojisiyle)
- PDF formatında indirilebilir
- Hikaye üretimi sürecinde, metin oluşturmak için Gemini API kullanılmıştır.

Bu sistem, öğrencilere hem okuma-yazma becerisi hem de karar verme pratiği kazandırmayı amaçlar.


## 📂 Proje Yapısı Özeti

| Dosya/Klasör | Açıklama |
|--------------|----------|
| `train_qlora_gemma_final_aggressive.py` | Gemma-2‑9B‑IT‑TR modeli için agresif parametrelerle LoRA üzerinden fine‑tuning scripti |
| `test_qlora_final.py` | Eğitilmiş modelin soru-cevap yeteneklerini test eden betik |
| `advanced_rag_fixed.py`, `advanced_rag.py` | Embedding — Retriever — Response Generation hattını içeren RAG pipeline |
| `advanced_rag_chat_fixed.py` | RAG chatbot versiyonu; kullanıcı mesajı alıp cevap üretir |
| `app.py`, `app4.py` | Flask tabanlı sohbet uygulama arayüzü |
| `requirements_qlora.txt` | Model eğitimi ve inference için gerekli kütüphaneler |
| `start_*.bat` | Windows’ta eğitim ve test komut dosyaları |
| `templates/`, `static/` | Web uygulaması ön yüz bileşenleri (HTML, CSS, JS) |
| `README_QLORA.md` | QLoRA özel konfigürasyonlarının detayları |

---
