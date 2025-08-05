# 🧠 Chatbot sistemi – `neuralwork/gemma-2-9b-it-tr` + QLoRA + RAG

Bu proje, **Hugging Face'teki `neuralwork/gemma-2-9b-it-tr`** modelini baz alır. Türkçe içeriklere özel LoRA tabanlı fine-tuning ve Retrieval‑Augmented Generation (RAG) ile belge destekli sohbet sistemi geliştirilmiştir.

---

## 🧪 Temel Model: `neuralwork/gemma-2-9b-it-tr`

- `google/gemma-2-9b-it` üzerine **yaklaşık 55k soru-cevap ve konuşma örneği ile fine-tune edilmiş** bir modeldir :contentReference[oaicite:1]{index=1}.
- LoRA parametreleri: `rank=128`, `lora_alpha=64`; eğitim süresi ~4 gün, RTX 6000 Ada GPU kullanılmıştır :contentReference[oaicite:2]{index=2}.
- Model, Türkçe'de daha iyi konuşma ve mantıksal çözümleme yeteneklerine sahiptir.

- Bu model LoRa ile fine tune edildi ve konu anlatımı yapabilen chatbot oluşturuldu. Bu aşamada kullanılan veri seti ve oluşan modelin linki aşağıda yer almaktadır.
 Model linki: https://huggingface.co/yagiz1323/EducationalChatbot/tree/main
 Model için kullanılan veri seti: https://huggingface.co/datasets/yagiz1323/FinetuningTurkishMat/tree/main

---

## 📂 Proje Yapısı Özeti

| Dosya/Klasör | Açıklama |
|--------------|----------|
| `train_qlora_gemma_final_aggressive.py` | Gemma-2‑9B‑IT‑TR modeli için agresif parametrelerle LoRA üzerinden fine‑tuning scripti |
| `test_qlora_final.py` | Eğitilmiş modelin soru-cevap yeteneklerini test eden betik |
| `advanced_rag_fixed.py` | Embedding — Retriever — Response Generation hattını içeren RAG pipeline |
| `advanced_rag_chat_fixed.py` | RAG chatbot versiyonu; kullanıcı mesajı alıp cevap üretir |
| `app.py`, `app4.py` | Flask tabanlı sohbet uygulama arayüzü |
| `requirements_qlora.txt` | Model eğitimi ve inference için gerekli kütüphaneler |
| `start_*.bat` | Windows’ta eğitim ve test komut dosyaları |
| `templates/`, `static/` | Web uygulaması ön yüz bileşenleri (HTML, CSS, JS) |
| `README_QLORA.md` | QLoRA özel konfigürasyonlarının detayları |

---
