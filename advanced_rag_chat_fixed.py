import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from advanced_rag_fixed import AdvancedMatematikRAG
import logging
from typing import Dict, Any, List
import numpy as np
import gc
import os
import re
import random

# CUDA memory optimizasyonları - RTX 4090 Laptop için optimize edilmiş
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Torch Dynamo compiler'ı devre dışı bırak
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# RTX 4090 Laptop için GPU optimizasyonları - ESKİ SİSTEME GÖRE
if torch.cuda.is_available():
    # GPU memory ayarları - ESKİ SİSTEME GÖRE
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.85)  # GPU'nun %85'ini kullan - GÜVENLİ
    torch.cuda.memory.empty_cache()
    gc.collect()

class AdvancedMatematikRAGChat:
    def __init__(self, model_path: str = "./gemma-2-9b-it-tr-new", rag_dir: str = "advanced_rag_system"):
        """
        Advanced Matematik RAG Chat sistemi - Gemma model ile optimize edilmiş
        
        Args:
            model_path: Gemma model path
            rag_dir: Advanced RAG sistemi dizini
        """
        self.model_path = model_path
        self.rag_dir = rag_dir
        
        # Device kontrolü
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Kullanılan cihaz: {self.device}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Bellek: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # GPU memory temizle
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()
            gc.collect()
        
        # Memory management arrays - 7 item limit
        self.user_messages = []  # Kullanıcı mesajları (max 7)
        self.bot_messages = []   # Bot mesajları (max 7)
        
        # Sınıf bilgisi ve konular
        self.user_class = None
        self.class_topics = {
            "1": [
                "📊 Sayılar (1-100)",
                "➕ Toplama İşlemi",
                "➖ Çıkarma İşlemi", 
                "🔢 Sayı Sıralaması",
                "📏 Uzunluk Ölçüleri",
                "⏰ Saat Okuma",
                "💰 Para Birimleri"
            ],
            "2": [
                "📊 Sayılar (1-1000)",
                "➕ Toplama İşlemi",
                "➖ Çıkarma İşlemi",
                "✖️ Çarpma İşlemi",
                "📏 Uzunluk Ölçüleri",
                "⏰ Saat ve Dakika",
                "💰 Para Hesaplama",
                "📐 Geometrik Şekiller"
            ],
            "3": [
                "📊 Sayılar (1-10000)",
                "➕ Toplama İşlemi",
                "➖ Çıkarma İşlemi",
                "✖️ Çarpma İşlemi",
                "➗ Bölme İşlemi",
                "📏 Uzunluk Ölçüleri",
                "⏰ Zaman Ölçüleri",
                "💰 Para Hesaplama",
                "📐 Geometrik Şekiller",
                "📊 Veri Toplama"
            ],
            "4": [
                "📊 Sayılar (1-100000)",
                "➕ Toplama İşlemi",
                "➖ Çıkarma İşlemi",
                "✖️ Çarpma İşlemi",
                "➗ Bölme İşlemi",
                "📏 Uzunluk Ölçüleri",
                "⏰ Zaman Ölçüleri",
                "💰 Para Hesaplama",
                "📐 Geometrik Şekiller",
                "📊 Veri Toplama",
                "🔢 Kesirler"
            ],
            "5": [
                "📊 Sayılar (1-1000000)",
                "➕ Toplama İşlemi",
                "➖ Çıkarma İşlemi",
                "✖️ Çarpma İşlemi",
                "➗ Bölme İşlemi",
                "📏 Uzunluk Ölçüleri",
                "⏰ Zaman Ölçüleri",
                "💰 Para Hesaplama",
                "📐 Geometrik Şekiller",
                "📊 Veri Toplama",
                "🔢 Kesirler",
                "📐 Alan Hesaplama"
            ],
            "6": [
                "📊 Sayılar (1-10000000)",
                "➕ Toplama İşlemi",
                "➖ Çıkarma İşlemi",
                "✖️ Çarpma İşlemi",
                "➗ Bölme İşlemi",
                "📏 Uzunluk Ölçüleri",
                "⏰ Zaman Ölçüleri",
                "💰 Para Hesaplama",
                "📐 Geometrik Şekiller",
                "📊 Veri Toplama",
                "🔢 Kesirler",
                "📐 Alan Hesaplama",
                "📊 Yüzde Hesaplama"
            ],
            "7": [
                "📊 Tam Sayılar",
                "➕ Toplama İşlemi",
                "➖ Çıkarma İşlemi",
                "✖️ Çarpma İşlemi",
                "➗ Bölme İşlemi",
                "📏 Uzunluk Ölçüleri",
                "⏰ Zaman Ölçüleri",
                "💰 Para Hesaplama",
                "📐 Geometrik Şekiller",
                "📊 Veri Toplama",
                "🔢 Kesirler",
                "📐 Alan Hesaplama",
                "📊 Yüzde Hesaplama",
                "📐 Çevre Hesaplama"
            ],
            "8": [
                "📊 Tam Sayılar",
                "➕ Toplama İşlemi",
                "➖ Çıkarma İşlemi",
                "✖️ Çarpma İşlemi",
                "➗ Bölme İşlemi",
                "📏 Uzunluk Ölçüleri",
                "⏰ Zaman Ölçüleri",
                "💰 Para Hesaplama",
                "📐 Geometrik Şekiller",
                "📊 Veri Toplama",
                "🔢 Kesirler",
                "📐 Alan Hesaplama",
                "📊 Yüzde Hesaplama",
                "📐 Çevre Hesaplama",
                "📊 Oran ve Orantı"
            ]
        }
        
        # Advanced RAG sistemi kontrolü ve otomatik eğitim
        self.check_and_train_rag_system()
        
        # Advanced RAG sistemi yükle
        print("🔄 Advanced RAG sistemi yükleniyor...")
        self.advanced_rag = AdvancedMatematikRAG()
        self.advanced_rag.load_advanced_rag_system(rag_dir)
        
        # LLM yükle - Gemma model için optimize edilmiş
        print("🔄 Gemma LLM model yükleniyor...")
        self.load_llm_safely()
        print("✅ Advanced RAG Chat sistemi başarıyla yüklendi!")
        
    def check_and_train_rag_system(self):
        """
        RAG sisteminin mevcut olup olmadığını kontrol et, yoksa eğit
        """
        if not os.path.exists(self.rag_dir):
            print("🔄 Advanced RAG sistemi bulunamadı, eğitim başlatılıyor...")
            self.train_rag_system()
        else:
            print("✅ Advanced RAG sistemi mevcut, yükleniyor...")
    
    def train_rag_system(self):
        """
        RAG sistemini eğit
        """
        try:
            # JSON dosyaları
            json_files = [
                "mat1.json",
                "mat2.json", 
                "mat3.json",
                "mat4.json",
                "mat5.json",
                "mat6.json",
                "mat7.json",
                "mat8.json",
                "mat8_lgs.json"
            ]
            
            # Mevcut dosyaları kontrol et
            existing_files = []
            for file in json_files:
                if os.path.exists(file):
                    existing_files.append(file)
                    print(f"✅ {file} dosyası bulundu")
                else:
                    print(f"⚠️  {file} dosyası bulunamadı, atlanıyor...")
            
            if not existing_files:
                print("❌ Hiçbir JSON dosyası bulunamadı! RAG sistemi eğitilemiyor.")
                return
            
            print(f"📚 {len(existing_files)} JSON dosyası bulundu. Advanced RAG sistemi eğitiliyor...")
            print("🔄 FOTOGRAFTEKİ SİSTEME GÖRE EĞİTİM BAŞLIYOR...")
            
            # Advanced RAG sistemi oluştur ve eğit
            advanced_rag = AdvancedMatematikRAG()
            advanced_rag.train_advanced_rag_system(existing_files, self.rag_dir)
            
            print("✅ Advanced RAG sistemi başarıyla eğitildi!")
            
        except Exception as e:
            print(f"❌ RAG sistemi eğitilirken hata oluştu: {e}")
            print("⚠️  Sistem LLM-only modunda çalışacak.")
        
    def load_llm_safely(self):
        """
        LLM'yi güvenli bir şekilde yükle - GPU memory sorunlarını çöz
        """
        try:
            # Önce GPU'da dene - DAHA FAZLA MEMORY KULLAN
            print("🔄 GPU'da LLM yükleniyor...")
            self.advanced_rag.load_llm()
            print("✅ LLM GPU'da başarıyla yüklendi!")
            
        except Exception as e:
            print(f"❌ GPU'da LLM yükleme hatası: {e}")
            
            # Memory temizle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.memory.empty_cache()
                gc.collect()
            
            # CPU'ya offload dene
            try:
                print("🔄 CPU'ya LLM yükleniyor...")
                self.advanced_rag.llm = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map={"": "cpu"},  # CPU'ya yükle
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                )
                self.advanced_rag.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                self.advanced_rag.llm.eval()
                print("✅ LLM CPU'ya yüklendi (GPU memory yetersiz)")
                
            except Exception as e2:
                print(f"❌ CPU yükleme de başarısız: {e2}")
                
                # Son çare: Daha küçük model ayarları
                try:
                    print("🔄 Son çare: Minimal ayarlarla yükleme...")
                    self.advanced_rag.llm = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        max_memory={0: "22GB"},  # ESKİ SİSTEME GÖRE
                        use_safetensors=True,
                        load_in_8bit=True  # 8-bit quantization
                    )
                    self.advanced_rag.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                    self.advanced_rag.llm.eval()
                    print("✅ LLM minimal ayarlarla yüklendi")
                    
                except Exception as e3:
                    print(f"❌ Tüm yükleme yöntemleri başarısız: {e3}")
                    raise
    
    def chat_with_llm_only(self, query):
        """
        Sadece LLM ile chat (RAG olmadan) - ESKİ SİSTEME GÖRE HIZLI
        
        Args:
            query: Kullanıcı sorgusu
            
        Returns:
            Bot yanıtı
        """
        try:
            # Advanced RAG sistemi kontrolü
            if not hasattr(self, 'advanced_rag') or self.advanced_rag is None:
                logger.error("Advanced RAG sistemi yüklenmemiş")
                return "Üzgünüm, sistem yüklenmemiş."
            
            # Gemma için düzeltilmiş prompt formatı
            prompt = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
            
            # LLM ile yanıt üret - HIZLI
            response = self.generate_with_llm_fast(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"LLM chat sırasında hata: {e}")
            return "Üzgünüm, yanıt üretirken bir hata oluştu."
    
    def generate_with_llm_fast(self, prompt: str, max_length: int = 30, temperature: float = 0.7) -> str:
        """
        LLM ile hızlı yanıt üret - ÇOK HIZLI VERSİYON
        
        Args:
            prompt: Input prompt
            max_length: Maksimum token uzunluğu (çok kısa)
            temperature: Sıcaklık parametresi
            
        Returns:
            Üretilen yanıt
        """
        try:
            # Advanced RAG sistemi kontrolü
            if not hasattr(self, 'advanced_rag') or self.advanced_rag is None:
                logger.error("Advanced RAG sistemi yüklenmemiş")
                return "Üzgünüm, sistem yüklenmemiş."
            
            # Tokenizer kontrolü
            if not hasattr(self.advanced_rag, 'tokenizer') or self.advanced_rag.tokenizer is None:
                logger.error("Tokenizer yüklenmemiş")
                return "Üzgünüm, tokenizer yüklenmemiş."
            
            # LLM kontrolü
            if not hasattr(self.advanced_rag, 'llm') or self.advanced_rag.llm is None:
                logger.error("LLM yüklenmemiş")
                return "Üzgünüm, LLM yüklenmemiş."
            
            # Tokenize - ÇOK KISA VE HIZLI
            inputs = self.advanced_rag.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)  # Biraz daha uzun
            
            # Device kontrolü - Model ve input aynı device'da olmalı
            model_device = next(self.advanced_rag.llm.parameters()).device
            if torch.cuda.is_available() and model_device.type == "cuda":
                for key in inputs:
                    inputs[key] = inputs[key].to(model_device)
            else:
                # Model CPU'da ise input'u da CPU'ya al
                for key in inputs:
                    inputs[key] = inputs[key].to("cpu")
            
            # Generate - ÇOK HIZLI AYARLAR
            with torch.no_grad():
                outputs = self.advanced_rag.llm.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=0.8,  # Daha düşük - hız için
                    top_k=5,     # Çok daha az - hız için
                    do_sample=True,
                    pad_token_id=self.advanced_rag.tokenizer.eos_token_id,
                    repetition_penalty=1.05,  # Daha düşük - hız için
                    num_beams=1,  # Greedy search - en hızlı
                    use_cache=True,  # Cache kullan - hız için
                    # early_stopping=True  # Bu parametre hataya neden oluyor
                )
            
            # Decode
            response = self.advanced_rag.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Temizle
            return self.advanced_rag.preprocess_response(response)
            
        except Exception as e:
            logger.error(f"LLM üretiminde hata: {e}")
            return "Üzgünüm, yanıt üretirken bir hata oluştu."
    
    def detect_class_from_message(self, message: str) -> str:
        """
        Mesajdan sınıf bilgisini tespit et - ESKİ SİSTEME GÖRE GELİŞTİRİLDİ
        
        Args:
            message: Kullanıcı mesajı
            
        Returns:
            Tespit edilen sınıf (1-8 arası) veya None
        """
        # Sınıf numaralarını ara - Çocukların doğal konuşma tarzları
        grade_patterns = [
            # Direkt sınıf belirtme - EN ÖNEMLİ OLANLAR ÖNCE
            r'(\d+)\s*(?:sınıf|sinif|grade|class)',
            r'(?:ben|benim)\s*(\d+)\s*(?:sınıf|sinif)',
            r'(\d+)\s*(?:inci|ıncı|uncu|üncü)\s*sınıf',
            r'(\d+)\s*(?:inci|ıncı|uncu|üncü)\s*sinif',
            
            # Doğal konuşma tarzları
            r'(?:ben|benim)\s*(\d+)\s*(?:sınıf|sinif)\s*(?:öğrencisi|ogrencisi)',
            r'(\d+)\s*(?:sınıf|sinif)\s*(?:öğrencisi|ogrencisi)',
            r'(?:ben|benim)\s*(\d+)\s*(?:sınıf|sinif)\s*(?:öğrencisiyim|ogrencisiyim)',
            r'(\d+)\s*(?:sınıf|sinif)\s*(?:öğrencisiyim|ogrencisiyim)',
            
            # Eğlenceli formatlar
            r'(?:ben|benim)\s*(\d+)\s*(?:sınıf|sinif)\s*(?:çocuğu|cocugu)',
            r'(\d+)\s*(?:sınıf|sinif)\s*(?:çocuğu|cocugu)',
            r'(?:ben|benim)\s*(\d+)\s*(?:sınıf|sinif)\s*(?:öğrencisi|ogrencisi)\s*(?:yim|im)',
            
            # Basit formatlar
            r'(\d+)\s*(?:sınıf|sinif)',
            r'(?:ben|benim)\s*(\d+)',
            r'(\d+)\s*(?:yaşındayım|yasindayim)',
            r'(\d+)\s*(?:yaş|yas)',
            r'(\d+)\s*(?:yaşında|yasinda)',
            r'(\d+)\s*(?:yaşında|yasinda)\s*(?:yim|im)',
            
            # Eğlenceli yaş formatları
            r'(?:ben|benim)\s*(\d+)\s*(?:yaş|yas)',
            r'(?:ben|benim)\s*(\d+)\s*(?:yaşında|yasinda)',
            r'(?:ben|benim)\s*(\d+)\s*(?:yaşında|yasinda)\s*(?:yim|im)',
            r'(?:ben|benim)\s*(\d+)\s*(?:yaşındayım|yasindayim)',
            
            # Ondalık yaş formatları
            r'(\d+\.\d+)\s*(?:yaş|yas)',
            r'(\d+\.\d+)\s*(?:yaşındayım|yasindayim)',
            r'(?:ben|benim)\s*(\d+\.\d+)\s*(?:yaş|yas)',
            r'(?:ben|benim)\s*(\d+\.\d+)\s*(?:yaşındayım|yasindayim)',
            
            # Eğlenceli ifadeler
            r'(?:ben|benim)\s*(\d+)\s*(?:yaşındayım|yasindayim)',
            r'(\d+)\s*(?:yaşındayım|yasindayim)',
            r'(?:ben|benim)\s*(\d+)\s*(?:yaş|yas)\s*(?:yim|im)',
            r'(\d+)\s*(?:yaş|yas)\s*(?:yim|im)',
            
            # Basit sayı formatları - EN SON
            r'(\d+)\s*(?:yaş|yas)',
            r'(?:ben|benim)\s*(\d+)',
        ]
        
        message_lower = message.lower()
        
        # TÜM PATTERN'LARI KONTROL ET VE EN İYİ EŞLEŞMEYİ BUL
        best_match = None
        best_score = 0
        
        for i, pattern in enumerate(grade_patterns):
            match = re.search(pattern, message_lower)
            if match:
                try:
                    grade = int(match.group(1))
                    if 1 <= grade <= 8:
                        # Pattern'ın önceliğini hesapla (daha spesifik = daha yüksek skor)
                        score = len(grade_patterns) - i  # İlk pattern'lar daha yüksek skor alır
                        
                        # Sınıf kelimesi varsa bonus puan
                        if 'sınıf' in pattern or 'sinif' in pattern:
                            score += 10
                        
                        # "ben" kelimesi varsa bonus puan
                        if 'ben' in pattern:
                            score += 5
                        
                        if score > best_score:
                            best_score = score
                            best_match = grade
                            
                except ValueError:
                    continue
        
        if best_match:
            return str(best_match)
        
        # Yaş belirtme formatlarını kontrol et (sadece sınıf bulunamadıysa)
        age_patterns = grade_patterns[15:]  # Sonraki pattern'lar yaş odaklı
        for pattern in age_patterns:
            match = re.search(pattern, message_lower)
            if match:
                age_str = match.group(1)
                try:
                    age = float(age_str)
                    # Yaştan sınıf hesaplama
                    grade = self.calculate_grade_from_age(age)
                    if grade:
                        return str(grade)
                except ValueError:
                    continue
        
        # Son çare: Sayısal değerleri kontrol et
        numbers = re.findall(r'\d+(?:\.\d+)?', message)
        for num_str in numbers:
            try:
                num = float(num_str)
                # Eğer 1-8 arası ise sınıf olarak kabul et
                if 1 <= num <= 8 and num.is_integer():
                    return str(int(num))
                # Eğer yaş aralığında ise yaştan hesapla
                elif 5 <= num <= 15:
                    grade = self.calculate_grade_from_age(num)
                    if grade:
                        return str(grade)
            except ValueError:
                continue
        
        return None
    
    def calculate_grade_from_age(self, age):
        """
        Yaştan sınıf hesapla - ESKİ SİSTEMDEN ALINDI
        
        Args:
            age: Yaş (float)
            
        Returns:
            Hesaplanan sınıf (1-8 arası) veya None
        """
        # Yaş-sınıf eşleştirmesi
        age_grade_mapping = {
            (5.0, 6.5): 1,    # 5-6.5 yaş = 1. sınıf
            (6.5, 7.5): 2,    # 6.5-7.5 yaş = 2. sınıf
            (7.5, 8.5): 3,    # 7.5-8.5 yaş = 3. sınıf
            (8.5, 9.5): 4,    # 8.5-9.5 yaş = 4. sınıf
            (9.5, 10.5): 5,   # 9.5-10.5 yaş = 5. sınıf
            (10.5, 11.5): 6,  # 10.5-11.5 yaş = 6. sınıf
            (11.5, 12.5): 7,  # 11.5-12.5 yaş = 7. sınıf
            (12.5, 13.5): 8   # 12.5-13.5 yaş = 8. sınıf
        }
        
        for (min_age, max_age), grade in age_grade_mapping.items():
            if min_age <= age <= max_age:
                return grade
        
        return None

    def get_grade_specific_files(self, grade):
        """
        Sınıfa özel dosya listesini döndür - ESKİ SİSTEMDEN ALINDI
        
        Args:
            grade: Sınıf numarası (1-8)
            
        Returns:
            Sınıfa özel dosya listesi
        """
        grade_files = {
            1: ["mat1.json"],
            2: ["mat2.json"],
            3: ["mat3.json"],
            4: ["mat4.json"],
            5: ["mat5.json"],
            6: ["mat6.json"],
            7: ["mat7.json"],
            8: ["mat8.json", "mat8_lgs.json"]
        }
        
        return grade_files.get(grade, [])

    def get_grade_topics(self, grade):
        """
        Sınıfa özel matematik konularını döndür - ESKİ SİSTEMDEN ALINDI
        
        Args:
            grade: Sınıf numarası (1-8)
            
        Returns:
            Sınıfa özel konu listesi
        """
        matematik_konulari = [
            # 1. SINIF
            [
                "Doğal Sayılar ve Sayma",
                "Nesne Sayısı Belirleme",
                "Sayıları Karşılaştırma ve Sıralama",
                "Toplama İşlemi",
                "Çıkarma İşlemi",
                "Geometri: Düzlemsel Şekiller",
                "Uzunluk Ölçme",
                "Tartma",
                "Zaman Ölçme (Saat, Gün, Ay)",
                "Para (TL ve kuruş)",
                "Veri Toplama ve Değerlendirme"
            ],
            # 2. SINIF
            [
                "Doğal Sayılar (1000'e kadar)",
                "Toplama ve Çıkarma İşlemleri",
                "Çarpma İşlemi",
                "Bölme İşlemi",
                "Geometri: Temel Düzlemsel ve Uzamsal Şekiller",
                "Uzunluk, Sıvı Ölçme",
                "Zaman Ölçme",
                "Para",
                "Veri Toplama ve Değerlendirme"
            ],
            # 3. SINIF
            [
                "Doğal Sayılar (10 000'e kadar)",
                "Dört İşlem",
                "Çarpanlar ve Katlar",
                "Kesirler",
                "Geometri: Doğru, Doğru Parçası ve Işın",
                "Açı ve Temel Geometrik Cisimler",
                "Zaman, Uzunluk, Sıvı Ölçme, Kütle",
                "Para",
                "Veri Toplama ve Grafik"
            ],
            # 4. SINIF
            [
                "Doğal Sayılar (1 000 000'a kadar)",
                "Dört İşlem Problemleri",
                "Kesirler",
                "Ondalık Gösterim",
                "Uzunluk, Alan, Hacim Ölçme",
                "Zaman Ölçme",
                "Geometri: Açılar, Dikdörtgen, Kare, Üçgen",
                "Simetri",
                "Veri Toplama ve Grafik"
            ],
            # 5. SINIF
            [
                "Doğal Sayılar ve Bölünebilme",
                "Asal Sayılar",
                "Kesirler",
                "Ondalık Kesirler",
                "Yüzdeler",
                "Geometri: Doğru, Doğru Parçası, Açılar",
                "Alan ve Çevre",
                "Hacim Ölçme",
                "Veri Analizi ve Olasılık",
                "Üslü ve Kareköklü Sayılar (Temel)"
            ],
            # 6. SINIF
            [
                "Doğal Sayılar ve Tam Sayılar",
                "Kesirler ve Ondalık Gösterim",
                "Oran-Orantı",
                "Yüzdeler",
                "Cebirsel İfadeler",
                "Denklemler",
                "Geometri: Çokgenler, Çember ve Daire",
                "Alan ve Hacim Ölçme",
                "Veri Analizi ve Olasılık"
            ],
            # 7. SINIF
            [
                "Rasyonel Sayılar",
                "Denklemler ve Eşitsizlikler",
                "Oran-Orantı ve Yüzde Problemleri",
                "Cebirsel İfadeler ve Özdeşlikler",
                "Geometri: Çokgenler, Dönüşüm Geometrisi",
                "Çember ve Daire",
                "Alan ve Hacim Problemleri",
                "Veri Analizi ve Olasılık"
            ],
            # 8. SINIF
            [
                "Çarpanlar ve Katlar",
                "Üslü İfadeler",
                "Kareköklü İfadeler",
                "Cebirsel İfadeler ve Özdeşlikler",
                "Denklem ve Eşitsizlikler",
                "Doğrusal Denklemler",
                "Eğim, Doğru Denklemi",
                "Geometri: Üçgenler, Dörtgenler, Çokgenler",
                "Geometrik Cisimler",
                "Olasılık",
                "Veri Analizi"
            ]
        ]
        
        if 1 <= grade <= 8:
            return matematik_konulari[grade - 1]
        return []

    def get_class_topics_response(self, class_num: str) -> str:
        """
        Sınıf konularını gösteren hazır cevap - ESKİ SİSTEME GÖRE GELİŞTİRİLDİ
        
        Args:
            class_num: Sınıf numarası
            
        Returns:
            Hazır cevap
        """
        # Sınıf numarasını integer'a çevir
        try:
            grade = int(class_num)
            grade_topics = self.get_grade_topics(grade)
            grade_files = self.get_grade_specific_files(grade)
        except ValueError:
            grade_topics = []
            grade_files = []
        
        topics_text = "\n".join([f"   {i+1}. {topic}" for i, topic in enumerate(grade_topics)])
        
        response = f"🎉 Harika! {class_num}. sınıf matematik dünyasına hoş geldin! 🐇"
        response += f"\n📚 Senin için {class_num}. sınıf matematik konularını hazırladım."
        response += f"\n🌟 Hangi konuda yardım istiyorsun? (örnek: toplama, çarpma, kesirler, geometri)"
        response += f"\n💡 Dosyalar: {', '.join(grade_files)}"
        
        # Konuları göster
        if grade_topics:
            response += f"\n📖 {class_num}. Sınıf Matematik Konuları:"
            response += f"\n{topics_text}"
        
        return response
    
    def generate_ratio_proportion_question(self, class_num: str) -> str:
        """
        Sınıfa uygun oran orantı sorusu üret (çözüm olmadan)
        
        Args:
            class_num: Sınıf numarası
            
        Returns:
            Oran orantı sorusu (çözüm olmadan)
        """
        try:
            grade = int(class_num)
        except ValueError:
            grade = 6  # Varsayılan 6. sınıf
        
        # Sınıfa göre oran orantı soruları
        questions = {
            6: [
                "Bir okulda 3 erkek öğrenciye karşılık 2 kız öğrenci vardır. Toplam 150 öğrenci olduğuna göre, kaç erkek öğrenci vardır?",
                "Bir işçi 8 saatte 24 parça üretiyor. Aynı hızda çalışarak 12 saatte kaç parça üretir?",
                "Bir araç 4 saatte 120 km yol alıyor. Aynı hızla 6 saatte kaç km yol alır?",
                "Bir bahçede 2 elma ağacına karşılık 3 armut ağacı vardır. Toplam 25 ağaç olduğuna göre, kaç elma ağacı vardır?"
            ],
            7: [
                "Bir bahçede elma ağaçlarının sayısı, armut ağaçlarının sayısının 2 katıdır. Toplam 45 ağaç olduğuna göre, kaç elma ağacı vardır?",
                "Bir araç 3 saatte 180 km yol alıyor. Aynı hızla 5 saatte kaç km yol alır?",
                "Bir işçi 5 saatte 30 parça üretiyor. Aynı hızda çalışarak 8 saatte kaç parça üretir?",
                "Bir sınıfta kız öğrencilerin sayısı erkek öğrencilerin sayısının 2/3'ü kadardır. Sınıfta 25 öğrenci olduğuna göre, kaç kız öğrenci vardır?"
            ],
            8: [
                "Bir sınıfta kız öğrencilerin sayısı erkek öğrencilerin sayısının 3/4'ü kadardır. Sınıfta 28 öğrenci olduğuna göre, kaç kız öğrenci vardır?",
                "Bir işçi 6 saatte 18 parça üretiyor. Aynı hızda çalışarak 10 saatte kaç parça üretir?",
                "Bir araç 2 saatte 80 km yol alıyor. Aynı hızla 7 saatte kaç km yol alır?",
                "Bir bahçede 3 gül çiçeğine karşılık 5 lale çiçeği vardır. Toplam 32 çiçek olduğuna göre, kaç gül çiçeği vardır?"
            ]
        }
        
        # Sınıfa uygun soruları al
        grade_questions = questions.get(grade, questions[6])  # Varsayılan 6. sınıf
        
        # Rastgele bir soru seç
        import random
        selected_question = random.choice(grade_questions)
        
        response = f"📊 {class_num}. Sınıf Oran Orantı Sorusu:\n\n"
        response += f"❓ **SORU:** {selected_question}\n\n"
        response += f"🎯 Bu soruyu çözmeye çalış! Cevabını öğrenmek istersen 'cevap' yazabilirsin."
        
        return response
    
    def generate_equation_inequality_question(self, class_num: str) -> str:
        """
        Sınıfa uygun denklem ve eşitsizlik sorusu üret (çözüm olmadan)
        
        Args:
            class_num: Sınıf numarası
            
        Returns:
            Denklem ve eşitsizlik sorusu
        """
        try:
            grade = int(class_num)
        except ValueError:
            grade = 8
        
        # Sınıfa göre denklem ve eşitsizlik soruları
        questions = {
            6: [
                "2x + 5 = 17 denklemini çözünüz.",
                "Bir sayının 3 katının 5 fazlası 23'tür. Bu sayıyı bulunuz.",
                "3x - 7 < 20 eşitsizliğini çözünüz.",
                "Bir sayının 2 katının 3 eksiği 15'ten küçüktür. Bu sayının alabileceği en büyük değeri bulunuz."
            ],
            7: [
                "3x + 8 = 26 denklemini çözünüz.",
                "Bir sayının 4 katının 7 fazlası 35'tir. Bu sayıyı bulunuz.",
                "5x - 12 ≤ 28 eşitsizliğini çözünüz.",
                "Bir sayının 3 katının 5 eksiği 25'ten büyük veya eşittir. Bu sayının alabileceği en küçük değeri bulunuz."
            ],
            8: [
                "4x + 15 = 47 denklemini çözünüz.",
                "Bir sayının 5 katının 8 fazlası 53'tür. Bu sayıyı bulunuz.",
                "6x - 18 < 42 eşitsizliğini çözünüz.",
                "Bir sayının 4 katının 6 eksiği 30'dan büyüktür. Bu sayının alabileceği en küçük değeri bulunuz.",
                "2x + 3y = 12 ve x - y = 2 denklem sistemini çözünüz."
            ]
        }
        
        # Sınıfa uygun soru seç
        if grade in questions:
            selected_question = random.choice(questions[grade])
        else:
            selected_question = random.choice(questions[8])  # Varsayılan 8. sınıf
        
        response = f"📊 {class_num}. Sınıf Denklem ve Eşitsizlik Sorusu:\n\n"
        response += f"❓ **SORU:** {selected_question}\n\n"
        response += f"🎯 Bu soruyu çözmeye çalış! Cevabını öğrenmek istersen 'cevap' yazabilirsin."
        
        return response
    
    def generate_rational_numbers_question(self, class_num: str) -> str:
        """
        Sınıfa uygun rasyonel sayılar sorusu üret (çözüm olmadan)
        
        Args:
            class_num: Sınıf numarası
            
        Returns:
            Rasyonel sayılar sorusu
        """
        try:
            grade = int(class_num)
        except ValueError:
            grade = 6
        
        # Sınıfa göre rasyonel sayılar soruları
        questions = {
            6: [
                "3/4 + 2/3 işleminin sonucu kaçtır?",
                "5/6 - 1/3 işleminin sonucu kaçtır?",
                "2/5 × 3/4 işleminin sonucu kaçtır?",
                "3/4 ÷ 2/3 işleminin sonucu kaçtır?"
            ],
            7: [
                "7/8 + 5/6 işleminin sonucu kaçtır?",
                "4/5 - 2/3 işleminin sonucu kaçtır?",
                "3/4 × 5/6 işleminin sonucu kaçtır?",
                "5/6 ÷ 3/4 işleminin sonucu kaçtır?",
                "2/3 + 1/4 - 1/6 işleminin sonucu kaçtır?"
            ],
            8: [
                "11/12 + 7/8 işleminin sonucu kaçtır?",
                "9/10 - 4/5 işleminin sonucu kaçtır?",
                "5/6 × 7/8 işleminin sonucu kaçtır?",
                "7/8 ÷ 5/6 işleminin sonucu kaçtır?",
                "3/4 + 2/3 - 1/2 işleminin sonucu kaçtır?"
            ]
        }
        
        # Sınıfa uygun soru seç
        if grade in questions:
            selected_question = random.choice(questions[grade])
        else:
            selected_question = random.choice(questions[6])  # Varsayılan 6. sınıf
        
        response = f"📊 {class_num}. Sınıf Rasyonel Sayılar Sorusu:\n\n"
        response += f"❓ **SORU:** {selected_question}\n\n"
        response += f"🎯 Bu soruyu çözmeye çalış! Cevabını öğrenmek istersen 'cevap' yazabilirsin."
        
        return response

    def generate_factors_multiples_question(self, class_num: str) -> str:
        """
        Çarpanlar ve Katlar sorusu üret (çözüm olmadan)
        
        Args:
            class_num: Sınıf numarası
            
        Returns:
            Çarpanlar ve Katlar sorusu
        """
        try:
            grade = int(class_num)
        except ValueError:
            grade = 6
        
        # Sınıfa göre çarpanlar ve katlar soruları
        questions = {
            3: [
                "12 sayısının çarpanları nelerdir?",
                "8 sayısının katları nelerdir? (ilk 5 tanesi)",
                "15 sayısının çarpanları nelerdir?",
                "6 sayısının katları nelerdir? (ilk 4 tanesi)"
            ],
            4: [
                "24 sayısının çarpanları nelerdir?",
                "10 sayısının katları nelerdir? (ilk 6 tanesi)",
                "18 sayısının çarpanları nelerdir?",
                "12 sayısının katları nelerdir? (ilk 5 tanesi)"
            ],
            5: [
                "36 sayısının çarpanları nelerdir?",
                "15 sayısının katları nelerdir? (ilk 5 tanesi)",
                "28 sayısının çarpanları nelerdir?",
                "20 sayısının katları nelerdir? (ilk 4 tanesi)"
            ],
            6: [
                "48 sayısının çarpanları nelerdir?",
                "18 sayısının katları nelerdir? (ilk 6 tanesi)",
                "42 sayısının çarpanları nelerdir?",
                "24 sayısının katları nelerdir? (ilk 5 tanesi)"
            ],
            7: [
                "60 sayısının çarpanları nelerdir?",
                "25 sayısının katları nelerdir? (ilk 5 tanesi)",
                "54 sayısının çarpanları nelerdir?",
                "30 sayısının katları nelerdir? (ilk 4 tanesi)"
            ],
            8: [
                "72 sayısının çarpanları nelerdir?",
                "32 sayısının katları nelerdir? (ilk 6 tanesi)",
                "66 sayısının çarpanları nelerdir?",
                "36 sayısının katları nelerdir? (ilk 5 tanesi)"
            ]
        }
        
        # Sınıfa uygun soru seç
        if grade in questions:
            selected_question = random.choice(questions[grade])
        else:
            selected_question = random.choice(questions[6])  # Varsayılan 6. sınıf
        
        response = f"📊 {class_num}. Sınıf Çarpanlar ve Katlar Sorusu:\n\n"
        response += f"❓ **SORU:** {selected_question}\n\n"
        response += f"🎯 Bu soruyu çözmeye çalış! Cevabını öğrenmek istersen 'cevap' yazabilirsin."
        
        return response
    
    def generate_general_math_question(self, class_num: str, query: str) -> str:
        """
        Genel matematik sorusu üret (RAG sistemi kullanarak)
        
        Args:
            class_num: Sınıf numarası
            query: Kullanıcı sorgusu
            
        Returns:
            Genel matematik sorusu
        """
        try:
            grade = int(class_num)
        except ValueError:
            grade = 6
        
        # RAG sistemi kullanarak soru üret
        try:
            if hasattr(self, 'advanced_rag') and self.advanced_rag is not None:
                # Konu bilgisini query'den çıkar
                import re
                konu_match = re.search(r'(\d+)\.\s*sınıf\s+(.+?)\s+konusundan', query)
                if konu_match:
                    konu_adi = konu_match.group(2).strip()
                    # RAG sistemi ile konuya özel soru üret
                    rag_query = f"{grade}. sınıf {konu_adi} konusundan soru üret"
                    relevant_docs = self.advanced_rag.retrieve_relevant_documents(rag_query, top_k=3)
                    
                    if relevant_docs:
                        # RAG'dan gelen bilgileri kullanarak soru üret
                        # Handle both document objects and dictionaries
                        context_parts = []
                        for doc in relevant_docs:
                            if hasattr(doc, 'page_content'):
                                context_parts.append(doc.page_content)
                            elif isinstance(doc, dict) and 'page_content' in doc:
                                context_parts.append(doc['page_content'])
                            elif isinstance(doc, dict) and 'content' in doc:
                                context_parts.append(doc['content'])
                            elif isinstance(doc, str):
                                context_parts.append(doc)
                        context = "\n".join(context_parts)
                        prompt = f"""
                        {grade}. sınıf {konu_adi} konusundan bir matematik sorusu üret.

                        Konu bilgileri:
                        {context}

                        Sadece soruyu üret, çözüm verme. Sadece soru metnini yaz, başka hiçbir şey yazma.
                        """
                        
                        response = self.generate_with_llm_fast(prompt, max_length=100, temperature=0.8)
                        
                        # Response'u temizle - sadece soru kısmını al
                        cleaned_response = self.extract_question_from_response(response)
                        if cleaned_response:
                            return cleaned_response
                        else:
                            # Fallback: Basit soru formatı
                            return f"{grade}. sınıf {konu_adi} konusundan bir soru: {response.strip()}"
                
                # Genel matematik sorusu üret
                rag_query = f"{grade}. sınıf matematik sorusu üret"
                relevant_docs = self.advanced_rag.retrieve_relevant_documents(rag_query, top_k=3)
                
                if relevant_docs:
                    # Handle both document objects and dictionaries
                    context_parts = []
                    for doc in relevant_docs:
                        if hasattr(doc, 'page_content'):
                            context_parts.append(doc.page_content)
                        elif isinstance(doc, dict) and 'page_content' in doc:
                            context_parts.append(doc['page_content'])
                        elif isinstance(doc, dict) and 'content' in doc:
                            context_parts.append(doc['content'])
                        elif isinstance(doc, str):
                            context_parts.append(doc)
                    context = "\n".join(context_parts)
                    prompt = f"""
                    {grade}. sınıf için bir matematik sorusu üret.

                    Konu bilgileri:
                    {context}

                    Sadece soruyu üret, çözüm verme. Sadece soru metnini yaz, başka hiçbir şey yazma.
                    """
                    
                    response = self.generate_with_llm_fast(prompt, max_length=100, temperature=0.8)
                    
                    # Response'u temizle - sadece soru kısmını al
                    cleaned_response = self.extract_question_from_response(response)
                    if cleaned_response:
                        return cleaned_response
                    else:
                        # Fallback: Basit soru formatı
                        return f"{grade}. sınıf için bir soru: {response.strip()}"
                    
        except Exception as e:
            logger.error(f"RAG soru üretme hatası: {e}")
        
        # Fallback: Sınıfa göre genel matematik soruları
        questions = {
            1: [
                "5 elma ve 3 armut var. Toplam kaç meyve var?",
                "8 kuş ve 2 kuş daha gelirse kaç kuş olur?",
                "10 kalem var, 4 tanesi kırıldı. Kaç kalem kaldı?",
                "3 kırmızı top ve 2 mavi top var. Toplam kaç top var?"
            ],
            2: [
                "15 + 8 = ? işlemini yap.",
                "24 - 7 = ? işlemini yap.",
                "3 x 4 = ? işlemini yap.",
                "12 ÷ 3 = ? işlemini yap."
            ],
            3: [
                "25 + 18 = ? işlemini yap.",
                "42 - 15 = ? işlemini yap.",
                "6 x 7 = ? işlemini yap.",
                "28 ÷ 4 = ? işlemini yap."
            ],
            4: [
                "156 + 89 = ? işlemini yap.",
                "234 - 67 = ? işlemini yap.",
                "8 x 9 = ? işlemini yap.",
                "72 ÷ 8 = ? işlemini yap."
            ],
            5: [
                "Bir dikdörtgenin uzun kenarı 6 cm, kısa kenarı 4 cm'dir. Bu dikdörtgenin alanı kaç cm²'dir?",
                "Bir karenin çevresi 20 cm'dir. Bu karenin alanı kaç cm²'dir?",
                "Bir üçgenin tabanı 5 cm, yüksekliği 3 cm'dir. Bu üçgenin alanı kaç cm²'dir?",
                "Bir dairenin yarıçapı 2 cm'dir. Bu dairenin çevresi kaç cm'dir? (π = 3 alınız)"
            ],
            6: [
                "Bir dikdörtgenin uzun kenarı 8 cm, kısa kenarı 5 cm'dir. Bu dikdörtgenin alanı kaç cm²'dir?",
                "Bir karenin çevresi 24 cm'dir. Bu karenin alanı kaç cm²'dir?",
                "Bir üçgenin tabanı 6 cm, yüksekliği 4 cm'dir. Bu üçgenin alanı kaç cm²'dir?",
                "Bir dairenin yarıçapı 3 cm'dir. Bu dairenin çevresi kaç cm'dir? (π = 3 alınız)"
            ],
            7: [
                "Bir dikdörtgen prizmanın boyutları 6 cm, 4 cm ve 3 cm'dir. Bu prizmanın hacmi kaç cm³'tür?",
                "Bir küpün bir ayrıtı 5 cm'dir. Bu küpün yüzey alanı kaç cm²'dir?",
                "Bir silindirin yarıçapı 4 cm, yüksekliği 6 cm'dir. Bu silindirin hacmi kaç cm³'tür? (π = 3 alınız)",
                "Bir üçgenin kenarları 5 cm, 12 cm ve 13 cm'dir. Bu üçgen dik üçgen midir?"
            ],
            8: [
                "Bir dikdörtgen prizmanın boyutları 8 cm, 6 cm ve 4 cm'dir. Bu prizmanın yüzey alanı kaç cm²'dir?",
                "Bir kürenin yarıçapı 3 cm'dir. Bu kürenin hacmi kaç cm³'tür? (π = 3 alınız)",
                "Bir koninin yarıçapı 5 cm, yüksekliği 12 cm'dir. Bu koninin hacmi kaç cm³'tür? (π = 3 alınız)",
                "Bir üçgenin kenarları 6 cm, 8 cm ve 10 cm'dir. Bu üçgen dik üçgen midir?"
            ]
        }
        
        # Sınıfa uygun soru seç
        if grade in questions:
            selected_question = random.choice(questions[grade])
        else:
            selected_question = random.choice(questions[6])  # Varsayılan 6. sınıf
        
        response = f"📊 {class_num}. Sınıf Matematik Sorusu:\n\n"
        response += f"❓ **SORU:** {selected_question}\n\n"
        response += f"🎯 Bu soruyu çözmeye çalış! Cevabını öğrenmek istersen 'cevap' yazabilirsin."
        
        return response
    
    def get_question_answer(self, class_num: str, question_text: str) -> str:
        """
        Sorunun çözümünü ver
        
        Args:
            class_num: Sınıf numarası
            question_text: Soru metni
            
        Returns:
            Sorunun çözümü
        """
        try:
            grade = int(class_num)
        except ValueError:
            grade = 6
        
        # Soru-cevap eşleştirmeleri
        solutions = {
            # 6. sınıf soruları
            "Bir okulda 3 erkek öğrenciye karşılık 2 kız öğrenci vardır. Toplam 150 öğrenci olduğuna göre, kaç erkek öğrenci vardır?": {
                "çözüm": "3x + 2x = 150\n5x = 150\nx = 30\nErkek öğrenci sayısı = 3x = 3 × 30 = 90",
                "cevap": "90 erkek öğrenci"
            },
            "Bir işçi 8 saatte 24 parça üretiyor. Aynı hızda çalışarak 12 saatte kaç parça üretir?": {
                "çözüm": "8 saat → 24 parça\n12 saat → x parça\n8x = 24 × 12\n8x = 288\nx = 36 parça",
                "cevap": "36 parça"
            },
            "Bir araç 4 saatte 120 km yol alıyor. Aynı hızla 6 saatte kaç km yol alır?": {
                "çözüm": "4 saat → 120 km\n6 saat → x km\n4x = 120 × 6\n4x = 720\nx = 180 km",
                "cevap": "180 km"
            },
            "Bir bahçede 2 elma ağacına karşılık 3 armut ağacı vardır. Toplam 25 ağaç olduğuna göre, kaç elma ağacı vardır?": {
                "çözüm": "2x + 3x = 25\n5x = 25\nx = 5\nElma ağacı sayısı = 2x = 2 × 5 = 10",
                "cevap": "10 elma ağacı"
            },
            
            # 7. sınıf soruları
            "Bir bahçede elma ağaçlarının sayısı, armut ağaçlarının sayısının 2 katıdır. Toplam 45 ağaç olduğuna göre, kaç elma ağacı vardır?": {
                "çözüm": "Armut: x, Elma: 2x\nx + 2x = 45\n3x = 45\nx = 15\nElma ağacı sayısı = 2x = 2 × 15 = 30",
                "cevap": "30 elma ağacı"
            },
            "Bir araç 3 saatte 180 km yol alıyor. Aynı hızla 5 saatte kaç km yol alır?": {
                "çözüm": "3 saat → 180 km\n5 saat → x km\n3x = 180 × 5\n3x = 900\nx = 300 km",
                "cevap": "300 km"
            },
            "Bir işçi 5 saatte 30 parça üretiyor. Aynı hızda çalışarak 8 saatte kaç parça üretir?": {
                "çözüm": "5 saat → 30 parça\n8 saat → x parça\n5x = 30 × 8\n5x = 240\nx = 48 parça",
                "cevap": "48 parça"
            },
            "Bir sınıfta kız öğrencilerin sayısı erkek öğrencilerin sayısının 2/3'ü kadardır. Sınıfta 25 öğrenci olduğuna göre, kaç kız öğrenci vardır?": {
                "çözüm": "Erkek: x, Kız: 2x/3\nx + 2x/3 = 25\n3x + 2x = 75\n5x = 75\nx = 15\nKız öğrenci = 2x/3 = 2×15/3 = 10",
                "cevap": "10 kız öğrenci"
            },
            
            # 8. sınıf soruları
            "Bir sınıfta kız öğrencilerin sayısı erkek öğrencilerin sayısının 3/4'ü kadardır. Sınıfta 28 öğrenci olduğuna göre, kaç kız öğrenci vardır?": {
                "çözüm": "Erkek: x, Kız: 3x/4\nx + 3x/4 = 28\n4x + 3x = 112\n7x = 112\nx = 16\nKız öğrenci = 3x/4 = 3×16/4 = 12",
                "cevap": "12 kız öğrenci"
            },
            "Bir işçi 6 saatte 18 parça üretiyor. Aynı hızda çalışarak 10 saatte kaç parça üretir?": {
                "çözüm": "6 saat → 18 parça\n10 saat → x parça\n6x = 18 × 10\n6x = 180\nx = 30 parça",
                "cevap": "30 parça"
            },
            "Bir araç 2 saatte 80 km yol alıyor. Aynı hızla 7 saatte kaç km yol alır?": {
                "çözüm": "2 saat → 80 km\n7 saat → x km\n2x = 80 × 7\n2x = 560\nx = 280 km",
                "cevap": "280 km"
            },
            "Bir bahçede 3 gül çiçeğine karşılık 5 lale çiçeği vardır. Toplam 32 çiçek olduğuna göre, kaç gül çiçeği vardır?": {
                "çözüm": "3x + 5x = 32\n8x = 32\nx = 4\nGül çiçeği sayısı = 3x = 3 × 4 = 12",
                "cevap": "12 gül çiçeği"
            },
            
            # Denklem ve eşitsizlik soruları
            "2x + 5 = 17 denklemini çözünüz.": {
                "çözüm": "2x + 5 = 17\n2x = 17 - 5\n2x = 12\nx = 6",
                "cevap": "x = 6"
            },
            "Bir sayının 3 katının 5 fazlası 23'tür. Bu sayıyı bulunuz.": {
                "çözüm": "x = sayı\n3x + 5 = 23\n3x = 23 - 5\n3x = 18\nx = 6",
                "cevap": "6"
            },
            "3x - 7 < 20 eşitsizliğini çözünüz.": {
                "çözüm": "3x - 7 < 20\n3x < 20 + 7\n3x < 27\nx < 9",
                "cevap": "x < 9"
            },
            "Bir sayının 2 katının 3 eksiği 15'ten küçüktür. Bu sayının alabileceği en büyük değeri bulunuz.": {
                "çözüm": "x = sayı\n2x - 3 < 15\n2x < 15 + 3\n2x < 18\nx < 9\nEn büyük tam sayı = 8",
                "cevap": "8"
            },
            "3x + 8 = 26 denklemini çözünüz.": {
                "çözüm": "3x + 8 = 26\n3x = 26 - 8\n3x = 18\nx = 6",
                "cevap": "x = 6"
            },
            "Bir sayının 4 katının 7 fazlası 35'tir. Bu sayıyı bulunuz.": {
                "çözüm": "x = sayı\n4x + 7 = 35\n4x = 35 - 7\n4x = 28\nx = 7",
                "cevap": "7"
            },
            "5x - 12 ≤ 28 eşitsizliğini çözünüz.": {
                "çözüm": "5x - 12 ≤ 28\n5x ≤ 28 + 12\n5x ≤ 40\nx ≤ 8",
                "cevap": "x ≤ 8"
            },
            "Bir sayının 3 katının 5 eksiği 25'ten büyük veya eşittir. Bu sayının alabileceği en küçük değeri bulunuz.": {
                "çözüm": "x = sayı\n3x - 5 ≥ 25\n3x ≥ 25 + 5\n3x ≥ 30\nx ≥ 10",
                "cevap": "10"
            },
            "4x + 15 = 47 denklemini çözünüz.": {
                "çözüm": "4x + 15 = 47\n4x = 47 - 15\n4x = 32\nx = 8",
                "cevap": "x = 8"
            },
            "Bir sayının 5 katının 8 fazlası 53'tür. Bu sayıyı bulunuz.": {
                "çözüm": "x = sayı\n5x + 8 = 53\n5x = 53 - 8\n5x = 45\nx = 9",
                "cevap": "9"
            },
            "6x - 18 < 42 eşitsizliğini çözünüz.": {
                "çözüm": "6x - 18 < 42\n6x < 42 + 18\n6x < 60\nx < 10",
                "cevap": "x < 10"
            },
            
            # Çarpanlar ve Katlar soruları
            "12 sayısının çarpanları nelerdir?": {
                "çözüm": "12'nin çarpanları: 1, 2, 3, 4, 6, 12\n12 = 1 × 12\n12 = 2 × 6\n12 = 3 × 4",
                "cevap": "1, 2, 3, 4, 6, 12"
            },
            "8 sayısının katları nelerdir? (ilk 5 tanesi)": {
                "çözüm": "8'in katları: 8, 16, 24, 32, 40\n8 × 1 = 8\n8 × 2 = 16\n8 × 3 = 24\n8 × 4 = 32\n8 × 5 = 40",
                "cevap": "8, 16, 24, 32, 40"
            },
            "15 sayısının çarpanları nelerdir?": {
                "çözüm": "15'in çarpanları: 1, 3, 5, 15\n15 = 1 × 15\n15 = 3 × 5",
                "cevap": "1, 3, 5, 15"
            },
            "6 sayısının katları nelerdir? (ilk 4 tanesi)": {
                "çözüm": "6'nın katları: 6, 12, 18, 24\n6 × 1 = 6\n6 × 2 = 12\n6 × 3 = 18\n6 × 4 = 24",
                "cevap": "6, 12, 18, 24"
            },
            "24 sayısının çarpanları nelerdir?": {
                "çözüm": "24'ün çarpanları: 1, 2, 3, 4, 6, 8, 12, 24\n24 = 1 × 24\n24 = 2 × 12\n24 = 3 × 8\n24 = 4 × 6",
                "cevap": "1, 2, 3, 4, 6, 8, 12, 24"
            },
            "10 sayısının katları nelerdir? (ilk 6 tanesi)": {
                "çözüm": "10'un katları: 10, 20, 30, 40, 50, 60\n10 × 1 = 10\n10 × 2 = 20\n10 × 3 = 30\n10 × 4 = 40\n10 × 5 = 50\n10 × 6 = 60",
                "cevap": "10, 20, 30, 40, 50, 60"
            },
            "18 sayısının çarpanları nelerdir?": {
                "çözüm": "18'in çarpanları: 1, 2, 3, 6, 9, 18\n18 = 1 × 18\n18 = 2 × 9\n18 = 3 × 6",
                "cevap": "1, 2, 3, 6, 9, 18"
            },
            "12 sayısının katları nelerdir? (ilk 5 tanesi)": {
                "çözüm": "12'nin katları: 12, 24, 36, 48, 60\n12 × 1 = 12\n12 × 2 = 24\n12 × 3 = 36\n12 × 4 = 48\n12 × 5 = 60",
                "cevap": "12, 24, 36, 48, 60"
            },
            "36 sayısının çarpanları nelerdir?": {
                "çözüm": "36'nın çarpanları: 1, 2, 3, 4, 6, 9, 12, 18, 36\n36 = 1 × 36\n36 = 2 × 18\n36 = 3 × 12\n36 = 4 × 9\n36 = 6 × 6",
                "cevap": "1, 2, 3, 4, 6, 9, 12, 18, 36"
            },
            "15 sayısının katları nelerdir? (ilk 5 tanesi)": {
                "çözüm": "15'in katları: 15, 30, 45, 60, 75\n15 × 1 = 15\n15 × 2 = 30\n15 × 3 = 45\n15 × 4 = 60\n15 × 5 = 75",
                "cevap": "15, 30, 45, 60, 75"
            },
            "28 sayısının çarpanları nelerdir?": {
                "çözüm": "28'in çarpanları: 1, 2, 4, 7, 14, 28\n28 = 1 × 28\n28 = 2 × 14\n28 = 4 × 7",
                "cevap": "1, 2, 4, 7, 14, 28"
            },
            "20 sayısının katları nelerdir? (ilk 4 tanesi)": {
                "çözüm": "20'nin katları: 20, 40, 60, 80\n20 × 1 = 20\n20 × 2 = 40\n20 × 3 = 60\n20 × 4 = 80",
                "cevap": "20, 40, 60, 80"
            },
            "48 sayısının çarpanları nelerdir?": {
                "çözüm": "48'in çarpanları: 1, 2, 3, 4, 6, 8, 12, 16, 24, 48\n48 = 1 × 48\n48 = 2 × 24\n48 = 3 × 16\n48 = 4 × 12\n48 = 6 × 8",
                "cevap": "1, 2, 3, 4, 6, 8, 12, 16, 24, 48"
            },
            "18 sayısının katları nelerdir? (ilk 6 tanesi)": {
                "çözüm": "18'in katları: 18, 36, 54, 72, 90, 108\n18 × 1 = 18\n18 × 2 = 36\n18 × 3 = 54\n18 × 4 = 72\n18 × 5 = 90\n18 × 6 = 108",
                "cevap": "18, 36, 54, 72, 90, 108"
            },
            "42 sayısının çarpanları nelerdir?": {
                "çözüm": "42'nin çarpanları: 1, 2, 3, 6, 7, 14, 21, 42\n42 = 1 × 42\n42 = 2 × 21\n42 = 3 × 14\n42 = 6 × 7",
                "cevap": "1, 2, 3, 6, 7, 14, 21, 42"
            },
            "24 sayısının katları nelerdir? (ilk 5 tanesi)": {
                "çözüm": "24'ün katları: 24, 48, 72, 96, 120\n24 × 1 = 24\n24 × 2 = 48\n24 × 3 = 72\n24 × 4 = 96\n24 × 5 = 120",
                "cevap": "24, 48, 72, 96, 120"
            },
            "Bir sayının 4 katının 6 eksiği 30'dan büyüktür. Bu sayının alabileceği en küçük değeri bulunuz.": {
                "çözüm": "x = sayı\n4x - 6 > 30\n4x > 30 + 6\n4x > 36\nx > 9\nEn küçük tam sayı = 10",
                "cevap": "10"
            },
            "2x + 3y = 12 ve x - y = 2 denklem sistemini çözünüz.": {
                "çözüm": "x - y = 2 → x = y + 2\n2(y + 2) + 3y = 12\n2y + 4 + 3y = 12\n5y + 4 = 12\n5y = 8\ny = 1.6\nx = 1.6 + 2 = 3.6",
                "cevap": "x = 3.6, y = 1.6"
            },
            
            # Rasyonel sayılar soruları
            "3/4 + 2/3 işleminin sonucu kaçtır?": {
                "çözüm": "3/4 + 2/3 = 9/12 + 8/12 = 17/12",
                "cevap": "17/12"
            },
            "5/6 - 1/3 işleminin sonucu kaçtır?": {
                "çözüm": "5/6 - 1/3 = 5/6 - 2/6 = 3/6 = 1/2",
                "cevap": "1/2"
            },
            "2/5 × 3/4 işleminin sonucu kaçtır?": {
                "çözüm": "2/5 × 3/4 = 6/20 = 3/10",
                "cevap": "3/10"
            },
            "3/4 ÷ 2/3 işleminin sonucu kaçtır?": {
                "çözüm": "3/4 ÷ 2/3 = 3/4 × 3/2 = 9/8",
                "cevap": "9/8"
            },
            "7/8 + 5/6 işleminin sonucu kaçtır?": {
                "çözüm": "7/8 + 5/6 = 21/24 + 20/24 = 41/24",
                "cevap": "41/24"
            },
            "4/5 - 2/3 işleminin sonucu kaçtır?": {
                "çözüm": "4/5 - 2/3 = 12/15 - 10/15 = 2/15",
                "cevap": "2/15"
            },
            "3/4 × 5/6 işleminin sonucu kaçtır?": {
                "çözüm": "3/4 × 5/6 = 15/24 = 5/8",
                "cevap": "5/8"
            },
            "5/6 ÷ 3/4 işleminin sonucu kaçtır?": {
                "çözüm": "5/6 ÷ 3/4 = 5/6 × 4/3 = 20/18 = 10/9",
                "cevap": "10/9"
            },
            "2/3 + 1/4 - 1/6 işleminin sonucu kaçtır?": {
                "çözüm": "2/3 + 1/4 - 1/6 = 8/12 + 3/12 - 2/12 = 9/12 = 3/4",
                "cevap": "3/4"
            },
            "11/12 + 7/8 işleminin sonucu kaçtır?": {
                "çözüm": "11/12 + 7/8 = 22/24 + 21/24 = 43/24",
                "cevap": "43/24"
            },
            "9/10 - 4/5 işleminin sonucu kaçtır?": {
                "çözüm": "9/10 - 4/5 = 9/10 - 8/10 = 1/10",
                "cevap": "1/10"
            },
            "5/6 × 7/8 işleminin sonucu kaçtır?": {
                "çözüm": "5/6 × 7/8 = 35/48",
                "cevap": "35/48"
            },
            "7/8 ÷ 5/6 işleminin sonucu kaçtır?": {
                "çözüm": "7/8 ÷ 5/6 = 7/8 × 6/5 = 42/40 = 21/20",
                "cevap": "21/20"
            },
            "3/4 + 2/3 - 1/2 işleminin sonucu kaçtır?": {
                "çözüm": "3/4 + 2/3 - 1/2 = 9/12 + 8/12 - 6/12 = 11/12",
                "cevap": "11/12"
            },
            
            # Genel matematik soruları
            "Bir dikdörtgenin uzun kenarı 8 cm, kısa kenarı 5 cm'dir. Bu dikdörtgenin alanı kaç cm²'dir?": {
                "çözüm": "Alan = uzun kenar × kısa kenar = 8 × 5 = 40 cm²",
                "cevap": "40 cm²"
            },
            "Bir karenin çevresi 24 cm'dir. Bu karenin alanı kaç cm²'dir?": {
                "çözüm": "Karenin bir kenarı = 24 ÷ 4 = 6 cm\nAlan = 6 × 6 = 36 cm²",
                "cevap": "36 cm²"
            },
            "Bir üçgenin tabanı 6 cm, yüksekliği 4 cm'dir. Bu üçgenin alanı kaç cm²'dir?": {
                "çözüm": "Alan = (taban × yükseklik) ÷ 2 = (6 × 4) ÷ 2 = 12 cm²",
                "cevap": "12 cm²"
            },
            "Bir dairenin yarıçapı 3 cm'dir. Bu dairenin çevresi kaç cm'dir? (π = 3 alınız)": {
                "çözüm": "Çevre = 2πr = 2 × 3 × 3 = 18 cm",
                "cevap": "18 cm"
            },
            "Bir dikdörtgen prizmanın boyutları 6 cm, 4 cm ve 3 cm'dir. Bu prizmanın hacmi kaç cm³'tür?": {
                "çözüm": "Hacim = uzunluk × genişlik × yükseklik = 6 × 4 × 3 = 72 cm³",
                "cevap": "72 cm³"
            },
            "Bir küpün bir ayrıtı 5 cm'dir. Bu küpün yüzey alanı kaç cm²'dir?": {
                "çözüm": "Yüzey alanı = 6 × (bir ayrıt)² = 6 × 5² = 6 × 25 = 150 cm²",
                "cevap": "150 cm²"
            },
            "Bir silindirin yarıçapı 4 cm, yüksekliği 6 cm'dir. Bu silindirin hacmi kaç cm³'tür? (π = 3 alınız)": {
                "çözüm": "Hacim = πr²h = 3 × 4² × 6 = 3 × 16 × 6 = 288 cm³",
                "cevap": "288 cm³"
            },
            "Bir üçgenin kenarları 5 cm, 12 cm ve 13 cm'dir. Bu üçgen dik üçgen midir?": {
                "çözüm": "5² + 12² = 25 + 144 = 169\n13² = 169\n5² + 12² = 13² olduğu için dik üçgendir",
                "cevap": "Evet, dik üçgendir"
            },
            "Bir dikdörtgen prizmanın boyutları 8 cm, 6 cm ve 4 cm'dir. Bu prizmanın yüzey alanı kaç cm²'dir?": {
                "çözüm": "Yüzey alanı = 2(ab + bc + ac) = 2(8×6 + 6×4 + 8×4) = 2(48 + 24 + 32) = 2×104 = 208 cm²",
                "cevap": "208 cm²"
            },
            "Bir kürenin yarıçapı 3 cm'dir. Bu kürenin hacmi kaç cm³'tür? (π = 3 alınız)": {
                "çözüm": "Hacim = (4/3)πr³ = (4/3) × 3 × 3³ = (4/3) × 3 × 27 = 108 cm³",
                "cevap": "108 cm³"
            },
            "Bir koninin yarıçapı 5 cm, yüksekliği 12 cm'dir. Bu koninin hacmi kaç cm³'tür? (π = 3 alınız)": {
                "çözüm": "Hacim = (1/3)πr²h = (1/3) × 3 × 5² × 12 = (1/3) × 3 × 25 × 12 = 300 cm³",
                "cevap": "300 cm³"
            },
            "Bir üçgenin kenarları 6 cm, 8 cm ve 10 cm'dir. Bu üçgen dik üçgen midir?": {
                "çözüm": "6² + 8² = 36 + 64 = 100\n10² = 100\n6² + 8² = 10² olduğu için dik üçgendir",
                "cevap": "Evet, dik üçgendir"
            }
        }
        
        # Sorunun çözümünü bul
        if question_text in solutions:
            solution = solutions[question_text]
            response = f"💡 **ÇÖZÜM:**\n{solution['çözüm']}\n\n"
            response += f"✅ **CEVAP:** {solution['cevap']}\n\n"
            response += f"🎉 Tebrikler! Soruyu çözmeye çalıştığın için çok güzel!"
        else:
            response = "Üzgünüm, bu sorunun çözümünü bulamadım. Başka bir soru sorabilir misin?"
        
        return response
    
    def extract_question_from_response(self, response: str) -> str:
        """
        Yanıttan soru metnini çıkar
        
        Args:
            response: Bot yanıtı
            
        Returns:
            Soru metni
        """
        # "❓ **SORU:** " kısmından sonrasını al
        if "❓ **SORU:** " in response:
            question_start = response.find("❓ **SORU:** ") + len("❓ **SORU:** ")
            question_end = response.find("\n\n", question_start)
            if question_end == -1:
                question_end = len(response)
            return response[question_start:question_end].strip()
        
        # Prompt instructions'ları temizle
        import re
        
        # Prompt instructions'ları kaldır
        patterns_to_remove = [
            r'\d+\.\s*sınıf\s+.*?konusundan\s+bir\s+matematik\s+sorusu\s+üret\.',
            r'Konu\s+bilgileri:',
            r'Sadece\s+soruyu\s+üret,\s+çözüm\s+verme\.',
            r'Sadece\s+soru\s+metnini\s+yaz,\s+başka\s+hiçbir\s+şey\s+yazma\.',
            r'\d+\.\s*sınıf\s+için\s+bir\s+matematik\s+sorusu\s+üret\.'
        ]
        
        cleaned_response = response
        for pattern in patterns_to_remove:
            cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.IGNORECASE | re.DOTALL)
        
        # Fazla boşlukları temizle
        cleaned_response = re.sub(r'\s+', ' ', cleaned_response).strip()
        
        # Eğer response hala prompt içeriyorsa, sadece soru kısmını al
        if "?" in cleaned_response:
            # Soru işaretine kadar olan kısmı al
            parts = cleaned_response.split("?")
            if len(parts) > 1:
                question_part = parts[0] + "?"
                return question_part.strip()
        
        # Eğer temizlenmiş response boşsa veya çok kısaysa, orijinal response'u döndür
        if len(cleaned_response) < 10:
            return response.strip()
        
        return cleaned_response
    
    def is_math_question(self, query: str) -> bool:
        """
        Sorunun matematik ile ilgili olup olmadığını kontrol et - ESKİ SİSTEME GÖRE HIZLI
        
        Args:
            query: Kullanıcı sorgusu
            
        Returns:
            True eğer matematik sorusu ise, False değilse
        """
        # Basit matematik kontrolü - ESKİ SİSTEME GÖRE
        math_keywords = ['toplama', 'çıkarma', 'çarpma', 'bölme', 'kesir', 'geometri', 'alan', 'çevre', 'hesapla', 'problem', 'soru']
        return any(keyword in query.lower() for keyword in math_keywords) or any(char.isdigit() for char in query)
    
    def chat(self, query: str, top_k: int = 5) -> str:
        """
        Advanced RAG chat fonksiyonu - ÇOK HIZLI VERSİYON
        
        Args:
            query: Kullanıcı sorgusu
            top_k: Döndürülecek doküman sayısı
            
        Returns:
            Model cevabı
        """
        try:
            # Özel durumlar için kontrol
            query_lower = query.lower()
            
            # Konu butonlarından gelen sorular - YENİ ÖZELLİK (EN ÜSTTE OLMALI)
            if any(keyword in query_lower for keyword in ['konusundan soru sorar mısın', 'konusundan soru sorar misin', 'konusundan soru', 'konusundan problem']):
                # Sınıf bilgisini query'den çıkar
                import re
                class_match = re.search(r'(\d+)\.\s*sınıf', query)
                if class_match:
                    detected_class = class_match.group(1)
                    self.user_class = detected_class
                    
                    # Konu adını çıkar
                    konu_match = re.search(r'(\d+)\.\s*sınıf\s+(.+?)\s+konusundan', query)
                    if konu_match:
                        konu_adi = konu_match.group(2).strip()
                        question = self.generate_general_math_question(detected_class, f"{detected_class}. sınıf {konu_adi} konusundan soru sorar mısın?")
                        # Son soruyu kaydet
                        self.last_question = self.extract_question_from_response(question)
                        return question
                
                # Eğer sınıf bilgisi yoksa, mevcut sınıfı kullan
                if self.user_class:
                    question = self.generate_general_math_question(self.user_class, query)
                    # Son soruyu kaydet
                    self.last_question = self.extract_question_from_response(question)
                    return question
                else:
                    return "Önce hangi sınıfta olduğunu söyler misin? Böylece sana uygun bir matematik sorusu sorabilirim."
            
            # Sınıf bilgisi kontrolü - HIZLI (KONU BUTONLARINDAN SONRA)
            detected_class = self.detect_class_from_message(query)
            if detected_class and not self.user_class:
                self.user_class = detected_class
                # Sadece hoş geldin mesajı ver, konuları listeleme
                return f"🎉 Harika! {detected_class}. sınıf matematik dünyasına hoş geldin! 🐇\n📚 Hangi konuda yardım istiyorsun? (örnek: toplama, çarpma, kesirler, geometri)"
            
            # Cevap isteme durumu - YENİ ÖZELLİK
            if any(keyword in query_lower for keyword in ['cevap', 'cevabi', 'çözüm', 'çözümü', 'sonuç', 'sonucu', 'kaç', 'kaçtır']):
                # Son soruyu bul
                if hasattr(self, 'last_question') and self.last_question:
                    return self.get_question_answer(self.user_class, self.last_question)
                else:
                    return "Henüz bir soru sormadım. Önce 'oran orantı sorar mısın?' diyerek bir soru iste!"
            
            # Oran orantı sorusu isteme durumu
            if any(keyword in query_lower for keyword in ['oran orantı', 'oran oranti', 'oran orantı sorar', 'oran oranti sorar']):
                if self.user_class:
                    question = self.generate_ratio_proportion_question(self.user_class)
                    # Son soruyu kaydet
                    self.last_question = self.extract_question_from_response(question)
                    return question
                else:
                    return "Önce hangi sınıfta olduğunu söyler misin? Böylece sana uygun bir oran orantı sorusu sorabilirim."
            
            # Denklem ve eşitsizlik sorusu isteme durumu - YENİ ÖZELLİK
            if any(keyword in query_lower for keyword in ['denklem', 'eşitsizlik', 'denklem ve eşitsizlik', 'denklem eşitsizlik']):
                if self.user_class:
                    question = self.generate_equation_inequality_question(self.user_class)
                    # Son soruyu kaydet
                    self.last_question = self.extract_question_from_response(question)
                    return question
                else:
                    return "Önce hangi sınıfta olduğunu söyler misin? Böylece sana uygun bir denklem ve eşitsizlik sorusu sorabilirim."
            
            # Rasyonel sayılar sorusu isteme durumu - YENİ ÖZELLİK
            if any(keyword in query_lower for keyword in ['rasyonel sayı', 'rasyonel sayilar', 'rasyonel sayılar', 'rasyonel']):
                if self.user_class:
                    question = self.generate_rational_numbers_question(self.user_class)
                    # Son soruyu kaydet
                    self.last_question = self.extract_question_from_response(question)
                    return question
                else:
                    return "Önce hangi sınıfta olduğunu söyler misin? Böylece sana uygun bir rasyonel sayılar sorusu sorabilirim."
            
            # Çarpanlar ve Katlar sorusu isteme durumu - YENİ ÖZELLİK
            if any(keyword in query_lower for keyword in ['çarpan', 'çarpanlar', 'kat', 'katlar', 'çarpanlar ve katlar', 'çarpan kat']):
                if self.user_class:
                    question = self.generate_factors_multiples_question(self.user_class)
                    # Son soruyu kaydet
                    self.last_question = self.extract_question_from_response(question)
                    return question
                else:
                    return "Önce hangi sınıfta olduğunu söyler misin? Böylece sana uygun bir çarpanlar ve katlar sorusu sorabilirim."
            

            
            # Genel matematik sorusu isteme durumu - YENİ ÖZELLİK
            if any(keyword in query_lower for keyword in ['sorar mısın', 'sorar misin', 'soru sorar', 'soru sor', 'problem sorar', 'problem sor']):
                if self.user_class:
                    question = self.generate_general_math_question(self.user_class, query)
                    # Son soruyu kaydet
                    self.last_question = self.extract_question_from_response(question)
                    return question
                else:
                    return "Önce hangi sınıfta olduğunu söyler misin? Böylece sana uygun bir matematik sorusu sorabilirim."
            
            # Basit matematik kontrolü - ÇOK HIZLI
            math_keywords = ['toplama', 'çıkarma', 'çarpma', 'bölme', 'kesir', 'geometri', 'alan', 'çevre', 'hesapla', 'problem', 'soru', 'oran', 'orantı', 'rasyonel']
            is_math = any(keyword in query_lower for keyword in math_keywords) or any(char.isdigit() for char in query)
            
            if not is_math:
                return self.chat_with_llm_only(query)
            
            # ÇOK HIZLI RAG PIPELINE - GEREKSİZ İŞLEMLERİ KALDIR
            try:
                # Advanced RAG sistemi kontrolü
                if not hasattr(self, 'advanced_rag') or self.advanced_rag is None:
                    logger.error("Advanced RAG sistemi yüklenmemiş")
                    return self.chat_with_llm_only(query)
                
                # Sadece 1 doküman al - ÇOK DAHA HIZLI
                relevant_docs = self.advanced_rag.retrieve_relevant_documents(query, top_k=1)
                
                if not relevant_docs:
                    logger.info("Doküman bulunamadı, LLM ile cevap üretiliyor...")
                    return self.chat_with_llm_only(query)
                
                # Basit prompt oluştur - ÇOK HIZLI
                context_prompt = self.advanced_rag.create_context_prompt(query, relevant_docs, max_context_length=500)  # Daha kısa
                
                # Hızlı cevap üret - ÇOK KISA
                response = self.generate_with_llm_fast(context_prompt, max_length=50, temperature=0.7)  # Daha kısa
                
                if response and response.strip():
                    return response
                else:
                    logger.info("RAG cevabı boş, LLM ile cevap üretiliyor...")
                    return self.chat_with_llm_only(query)
                    
            except Exception as rag_error:
                logger.error(f"RAG pipeline hatası: {rag_error}")
                return self.chat_with_llm_only(query)
            
        except Exception as e:
            logger.error(f"Chat sırasında hata: {e}")
            return "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin."
    
    def add_to_memory(self, user_message: str, bot_message: str):
        """
        Memory'ye mesaj ekle (7 item limit - FOTOGRAFTEKİ SİSTEME GÖRE)
        
        Args:
            user_message: Kullanıcı mesajı
            bot_message: Bot mesajı
        """
        # Kullanıcı mesajını ekle
        self.user_messages.append(user_message)
        if len(self.user_messages) > 7:
            self.user_messages.pop(0)
        
        # Bot mesajını ekle
        self.bot_messages.append(bot_message)
        if len(self.bot_messages) > 7:
            self.bot_messages.pop(0)
    
    def get_memory_context(self) -> str:
        """
        Memory'den context oluştur
        
        Returns:
            Memory context
        """
        if not self.user_messages:
            return ""
        
        context = "Önceki konuşma:\n"
        for i in range(len(self.user_messages)):
            context += f"Kullanıcı: {self.user_messages[i]}\n"
            if i < len(self.bot_messages):
                context += f"Bot: {self.bot_messages[i]}\n"
        context += "\nŞimdi sadece bu soruya cevap ver:\n"
        
        return context
    
    def interactive_chat(self):
        """İnteraktif chat modu - ÇOK HIZLI VERSİYON"""
        # Açılış mesajı
        print("🐇🌟 Hoş geldin minik keşifçi! Ben minik tavşan yoldaşın. Matematik dünyasında senin rehberin olacağım! Hadi bana kaçıncı sınıfa gittiğini söyle, birlikte başlayalım! 🎒🥳")
        print()
        
        while True:
            try:
                # Input alma - DAHA GÜVENLİ
                try:
                    user_input = input("👤 Sen: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 Görüşmek üzere!")
                    break
                except Exception as input_error:
                    logger.error(f"Input alma hatası: {input_error}")
                    print("🤖 Bot: Input alma hatası. Lütfen tekrar deneyin.")
                    continue
                
                # Çıkış kontrolü
                if user_input.lower() in ["exit", "quit", "çıkış", "çıkıs"]:
                    print("👋 Görüşmek üzere!")
                    break
                
                # Boş input kontrolü
                if not user_input:
                    continue
                
                # Chat fonksiyonunu çağır - ÇOK HIZLI
                print("🔄 Yanıt üretiliyor...")
                response = self.chat(user_input)
                
                if response and response.strip():
                    print("🤖 Bot:", response)
                    self.add_to_memory(user_input, response)
                else:
                    print("🤖 Bot: Yanıt üretilemedi. Lütfen tekrar deneyin.")
                    
            except KeyboardInterrupt:
                print("\n👋 Görüşmek üzere!")
                break
            except EOFError:
                print("\n👋 Program sonlandırıldı.")
                break
            except Exception as e:
                logger.error(f"Chat sırasında hata: {e}")
                print("🤖 Bot: Bir hata oluştu. Lütfen tekrar deneyin.")
                continue

def main():
    """Ana fonksiyon - GÜVENLİ VERSİYON"""
    chat_system = None
    try:
        # Advanced RAG Chat sistemi oluştur
        print("🚀 Advanced RAG Chat sistemi başlatılıyor...")
        chat_system = AdvancedMatematikRAGChat()
        
        # İnteraktif chat başlat
        chat_system.interactive_chat()
        
    except KeyboardInterrupt:
        print("\n👋 Program kullanıcı tarafından sonlandırıldı.")
    except EOFError:
        print("\n👋 Program sonlandırıldı.")
    except Exception as e:
        logger.error(f"Program başlatılırken hata: {e}")
        print(f"❌ Program başlatılamadı: {e}")
        print("💡 Lütfen Advanced RAG sisteminin eğitildiğinden emin olun.")
        
        # Hata durumunda da chat'i dene
        if chat_system:
            try:
                print("🔄 Chat sistemi tekrar deneniyor...")
                chat_system.interactive_chat()
            except Exception as retry_error:
                print(f"❌ Chat sistemi de başarısız: {retry_error}")
    finally:
        # Program sonunda GPU belleğini temizle
        try:
            print("🧹 GPU belleği temizleniyor...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.memory.empty_cache()
                gc.collect()
                print("✅ GPU belleği temizlendi!")
        except Exception as cleanup_error:
            print(f"⚠️ Bellek temizleme hatası: {cleanup_error}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Program kullanıcı tarafından sonlandırıldı.")
    except EOFError:
        print("\n👋 Program sonlandırıldı.")
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {e}")
    finally:
        print("👋 Program sonlandırıldı.")

# main() çağrısı yukarıda yapıldı 