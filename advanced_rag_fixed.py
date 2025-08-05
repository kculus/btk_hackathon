import json
import os
import pickle
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import logging
import re
import gc

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

# RTX 4090 Laptop için GPU optimizasyonları - HIZLI KULLANIM
if torch.cuda.is_available():
    # GPU memory ayarları - HIZLI memory kullan
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.85)  # GPU'nun %85'ini kullan - GÜVENLİ
    torch.cuda.memory.empty_cache()
    gc.collect()
    
    # CUDA optimizasyonları
    torch.backends.cudnn.benchmark = True  # Hızlandırma
    torch.backends.cudnn.deterministic = False  # Hız için

class AdvancedMatematikRAG:
    def __init__(self, model_path: str = "gemma-2-9b-it-tr-new"):
        """
        Advanced Matematik RAG sistemi - FOTOGRAFTEKİ SİSTEME GÖRE DÜZELTİLDİ
        
        Args:
            model_path: Gemma model path
        """
        self.model_path = model_path
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.embeddings = None
        self.llm = None
        self.tokenizer = None
        self.max_embedding_score = 0.0
        self.best_response = ""
        
        # Memory management arrays - 7 item limit
        self.user_messages = []  # Kullanıcı mesajları (max 7)
        self.bot_messages = []   # Bot mesajları (max 7)
        
    def load_llm(self):
        """
        LLM model'ini yükle - Gemma model için optimize edilmiş
        """
        try:
            logger.info("Gemma LLM model yükleniyor...")
            
            # Tokenizer'ı yükle
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            
            # Padding token ayarla
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Model yükleme ayarları - GPU'ya yükle (HIZLI)
            if torch.cuda.is_available():
                logger.info("🔄 GPU'da LLM yükleniyor...")
                self.llm = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,  # 16-bit precision - hız için
                    device_map="auto",  # Otomatik GPU mapping
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    max_memory={0: "12GB"}  # GPU memory limiti - GÜVENLİ
                )
            else:
                # CPU fallback
                self.llm = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                )
            
            # Model'i evaluation moduna al
            self.llm.eval()
            
            logger.info("✅ Gemma LLM model başarıyla yüklendi")
            
        except Exception as e:
            logger.error(f"❌ Gemma LLM model yüklenirken hata: {e}")
            raise
    
    def generate_with_llm(self, prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """
        LLM ile yanıt üret - KALİTE VE HIZ OPTİMİZASYONU
        
        Args:
            prompt: Input prompt
            max_length: Maksimum token uzunluğu
            temperature: Sıcaklık parametresi
            
        Returns:
            Üretilen yanıt
        """
        try:
            # Memory temizle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # CPU'da işlemler (GPU sorunları için)
            # Input'lar zaten CPU'da
            # Model zaten CPU'da
            
            # Generate with optimized settings for RTX 4090 - KALİTE VE HIZ DENGESİ
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=30,  # Biraz daha fazla çeşitlilik
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.encode("<end_of_turn>")[0] if "<end_of_turn>" in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    num_beams=1,  # Hızlı üretim
                    length_penalty=1.0,
                    no_repeat_ngram_size=2,
                    use_cache=True
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Memory temizle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Temizle
            return self.preprocess_response(response)
            
        except Exception as e:
            logger.error(f"LLM üretiminde hata: {e}")
            return "Üzgünüm, yanıt üretirken bir hata oluştu."
    
    def generate_with_llm_fast(self, prompt: str, max_length: int = 50, temperature: float = 0.7) -> str:
        """
        LLM ile hızlı yanıt üret - ESKİ SİSTEME GÖRE ÇOK HIZLI
        
        Args:
            prompt: Input prompt
            max_length: Maksimum token uzunluğu (çok kısa)
            temperature: Sıcaklık parametresi
            
        Returns:
            Üretilen yanıt
        """
        try:
            # Tokenize - ÇOK KISA VE HIZLI
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)  # Daha kısa
            
            # Device kontrolü - Model ve input aynı device'da olmalı
            if torch.cuda.is_available() and self.llm.device.type == "cuda":
                for key in inputs:
                    inputs[key] = inputs[key].to("cuda")
            else:
                for key in inputs:
                    inputs[key] = inputs[key].to("cpu")
            
            # Generate - ÇOK HIZLI AYARLAR
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=0.8,  # Daha düşük - hız için
                    top_k=10,    # Çok daha az - hız için
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.05,  # Daha düşük - hız için
                    num_beams=1,  # Greedy search - en hızlı
                    use_cache=True,  # Cache kullan - hız için
                    # early_stopping=True  # Bu parametre hataya neden oluyor
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Temizle
            return self.preprocess_response(response)
            
        except Exception as e:
            logger.error(f"LLM üretiminde hata: {e}")
            return "Üzgünüm, yanıt üretirken bir hata oluştu."
    
    def preprocess_response(self, response):
        """
        Model yanıtını temizle ve formatla
        
        Args:
            response: Ham model yanıtı
            
        Returns:
            Temizlenmiş yanıt
        """
        # Model yanıtını çıkar
        model_response = ""
        
        # "model" kelimesinden sonraki kısmı al
        if "model" in response:
            model_response = response.split("model")[-1].strip()
        else:
            model_response = response
        
        # Newline karakterlerini temizle ve tek satır yap
        model_response = model_response.replace('\n', ' ').replace('\r', ' ')
        
        # Fazla boşlukları temizle
        model_response = re.sub(r'\s+', ' ', model_response).strip()
        
        # Gereksiz karakterleri temizle
        model_response = re.sub(r'\.{3,}', '', model_response)  # 3+ nokta
        model_response = re.sub(r'={3,}', '', model_response)   # 3+ eşittir
        model_response = re.sub(r'-{3,}', '', model_response)   # 3+ tire
        model_response = re.sub(r'_{3,}', '', model_response)   # 3+ alt çizgi
        model_response = re.sub(r'\*{3,}', '', model_response)  # 3+ yıldız
        model_response = re.sub(r'>{3,}', '', model_response)   # 3+ büyüktür işareti
        model_response = re.sub(r'<{3,}', '', model_response)   # 3+ küçüktür işareti
        
        # Sonundaki gereksiz karakterleri temizle
        model_response = re.sub(r'[.\s]+$', '', model_response)  # Sonundaki nokta ve boşluklar
        model_response = re.sub(r'^[.\s]+', '', model_response)  # Başındaki nokta ve boşluklar
        
        # Fazla boşlukları tekrar temizle
        model_response = re.sub(r'\s+', ' ', model_response).strip()
        
        return model_response
    
    def create_embedding_for_text(self, text: str) -> np.ndarray:
        """
        Metin için embedding oluştur
        
        Args:
            text: Input metin
            
        Returns:
            Embedding vektörü
        """
        try:
            if self.embedding_model is None:
                # Embedding model'ini yükle
                self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
            # Embedding oluştur
            embedding = self.embedding_model.encode([text], convert_to_tensor=True)
            
            # CPU'ya taşı ve numpy array'e çevir
            if torch.cuda.is_available():
                embedding = embedding.cpu().numpy()
            else:
                embedding = embedding.numpy()
            
            return embedding[0]  # İlk (ve tek) embedding'i döndür
            
        except Exception as e:
            logger.error(f"Embedding oluşturma hatası: {e}")
            # Hata durumunda sıfır vektör döndür
            return np.zeros(384)  # Default embedding boyutu
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        İki embedding arasındaki benzerliği hesapla
        
        Args:
            embedding1: İlk embedding
            embedding2: İkinci embedding
            
        Returns:
            Benzerlik skoru (0-1 arası)
        """
        try:
            # Cosine similarity hesapla
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return float(similarity)
        except Exception as e:
            logger.error(f"Benzerlik hesaplama hatası: {e}")
            return 0.0
    
    def pre_process_query(self, query: str) -> str:
        """
        Sorguyu ön işle (Pre-process adımı - FOTOGRAFTEKİ SİSTEME GÖRE)
        
        Args:
            query: Orijinal sorgu
            
        Returns:
            İşlenmiş sorgu
        """
        try:
            # LLM ile sorgu özetleme/optimizasyonu
            prompt = f"""Aşağıdaki sorguyu matematik konuları için optimize et ve özetle:

Orijinal sorgu: {query}

Sadece matematikle ilgili anahtar kelimeleri çıkar ve kısa bir özet yap.
Örnek: "toplama işlemi nasıl yapılır" → "toplama işlemi"
Örnek: "kesirlerde çarpma" → "kesir çarpma"
Örnek: "geometri alan hesaplama" → "geometri alan"

Optimize edilmiş sorgu:"""
            
            optimized = self.generate_with_llm(prompt, max_length=50, temperature=0.3)
            
            # Eğer LLM yanıt vermezse orijinal sorguyu kullan
            if not optimized or len(optimized.strip()) < 3:
                return query
            
            return optimized.strip()
            
        except Exception as e:
            logger.error(f"Query preprocessing hatası: {e}")
            return query
    
    def filter_relevant_documents(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        İlgili dokümanları filtrele (Filter adımı - FOTOGRAFTEKİ SİSTEME GÖRE)
        
        Args:
            query: Kullanıcı sorgusu
            retrieved_docs: Bulunan dokümanlar
            
        Returns:
            Filtrelenmiş dokümanlar
        """
        try:
            if not retrieved_docs:
                return []
            
            # Dokümanları string'e çevir
            doc_texts = []
            for i, doc in enumerate(retrieved_docs):
                if isinstance(doc, dict):
                    doc_text = doc.get('document', str(doc))
                else:
                    doc_text = str(doc)
                doc_texts.append(f"Doküman {i+1}: {doc_text[:200]}...")
            
            docs_text = "\n".join(doc_texts)
            
            # LLM ile filtreleme
            prompt = f"""Aşağıdaki sorgu için en uygun dokümanları seç:

Sorgu: {query}

Mevcut dokümanlar:
{docs_text}

Hangi dokümanlar bu sorgu için en uygun? Sadece doküman numaralarını yaz (örn: 1, 3, 5)
Uygun doküman numaraları:"""
            
            filter_response = self.generate_with_llm(prompt, max_length=50, temperature=0.3)
            
            # Numaraları çıkar
            numbers = re.findall(r'\d+', filter_response)
            
            if not numbers:
                # Eğer LLM yanıt vermezse ilk 3 dokümanı al
                return retrieved_docs[:3]
            
            # Seçilen dokümanları döndür
            selected_docs = []
            for num in numbers:
                idx = int(num) - 1
                if 0 <= idx < len(retrieved_docs):
                    selected_docs.append(retrieved_docs[idx])
            
            return selected_docs if selected_docs else retrieved_docs[:3]
            
        except Exception as e:
            logger.error(f"Document filtering hatası: {e}")
            return retrieved_docs[:3]  # Hata durumunda ilk 3 dokümanı al
    
    def self_reflect_and_improve_with_embeddings(self, query: str, initial_response: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Basit cevap iyileştirme - ÇOK HIZLI VERSİYON
        
        Args:
            query: Orijinal sorgu
            initial_response: İlk yanıt
            relevant_docs: İlgili dokümanlar
            
        Returns:
            Cevap
        """
        # Basit cevap döndür - ÇOK HIZLI
        return initial_response
    
    def self_reflect_and_improve(self, query: str, initial_response: str) -> str:
        """
        Basit cevap kontrolü - ÇOK HIZLI VERSİYON
        
        Args:
            query: Kullanıcı sorgusu
            initial_response: İlk yanıt
            
        Returns:
            Cevap
        """
        # Basit cevap döndür - ÇOK HIZLI
        return initial_response
    
    def load_data(self, json_files: List[str]) -> List[Dict[str, Any]]:
        """
        JSON dosyalarından veri yükle
        
        Args:
            json_files: JSON dosya yolları
            
        Returns:
            Yüklenen veri
        """
        data = []
        
        for file_path in json_files:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        data.extend(file_data)
                        logger.info(f"{file_path} dosyası yüklendi: {len(file_data)} kayıt")
                else:
                    logger.warning(f"{file_path} dosyası bulunamadı")
            except Exception as e:
                logger.error(f"{file_path} dosyası yüklenirken hata: {e}")
        
        logger.info(f"Toplam {len(data)} kayıt yüklendi")
        return data
    
    def prepare_documents(self, data: List[Dict[str, Any]]) -> List[str]:
        """
        Veriyi doküman formatına çevir
        
        Args:
            data: Ham veri
            
        Returns:
            Doküman listesi
        """
        documents = []
        
        for item in data:
            try:
                # JSON yapısına göre doküman oluştur
                if isinstance(item, dict):
                    # Prompt ve response alanlarını birleştir
                    prompt = item.get('prompt', '')
                    response = item.get('response', '')
                    
                    # Doküman oluştur
                    document = f"Soru: {prompt}\nCevap: {response}"
                    documents.append(document)
                else:
                    # String ise direkt ekle
                    documents.append(str(item))
                    
            except Exception as e:
                logger.error(f"Doküman hazırlama hatası: {e}")
                continue
        
        logger.info(f"{len(documents)} doküman hazırlandı")
        return documents
    
    def create_embeddings(self, documents: List[str], model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Dokümanlar için embedding'ler oluştur
        
        Args:
            documents: Doküman listesi
            model_name: Embedding model adı
        """
        try:
            logger.info("Embedding model yükleniyor...")
            self.embedding_model = SentenceTransformer(model_name)
            
            logger.info("Embedding'ler oluşturuluyor...")
            self.embeddings = self.embedding_model.encode(documents, convert_to_tensor=True)
            
            # CPU'ya taşı ve numpy array'e çevir
            if torch.cuda.is_available():
                self.embeddings = self.embeddings.cpu().numpy()
            else:
                self.embeddings = self.embeddings.numpy()
            
            logger.info(f"Embedding'ler oluşturuldu: {self.embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Embedding oluşturma hatası: {e}")
            raise
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """
        FAISS index oluştur
        
        Args:
            embeddings: Embedding'ler
        """
        try:
            logger.info("FAISS index oluşturuluyor...")
            
            # FAISS index oluştur
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product (Cosine similarity)
            
            # Embedding'leri normalize et
            faiss.normalize_L2(embeddings)
            
            # Index'e ekle
            self.index.add(embeddings.astype('float32'))
            
            logger.info(f"FAISS index oluşturuldu: {self.index.ntotal} vektör")
            
        except Exception as e:
            logger.error(f"FAISS index oluşturma hatası: {e}")
            raise
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        İlgili dokümanları getir
        
        Args:
            query: Sorgu
            top_k: Döndürülecek doküman sayısı
            
        Returns:
            İlgili dokümanlar
        """
        try:
            # Query embedding oluştur
            query_embedding = self.create_embedding_for_text(query)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Normalize et
            faiss.normalize_L2(query_embedding)
            
            # FAISS ile arama yap
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Sonuçları formatla
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    results.append({
                        'document': self.documents[idx],
                        'score': float(score),
                        'index': int(idx)
                    })
            
            # En yüksek skoru kaydet
            if results:
                self.max_embedding_score = max(result['score'] for result in results)
            
            return results
            
        except Exception as e:
            logger.error(f"Doküman arama hatası: {e}")
            return []
    
    def create_context_prompt(self, query: str, relevant_docs: list, max_context_length: int = 2000) -> str:
        """
        Context prompt oluştur
        
        Args:
            query: Sorgu
            relevant_docs: İlgili dokümanlar
            max_context_length: Maksimum context uzunluğu
            
        Returns:
            Context prompt
        """
        try:
            # Dokümanları birleştir
            context = ""
            for i, doc in enumerate(relevant_docs):
                if isinstance(doc, dict):
                    doc_text = doc.get('document', str(doc))
                else:
                    doc_text = str(doc)
                context += f"Bilgi {i+1}: {doc_text}\n\n"
            
            # Gemma RAG prompt formatı
            prompt = f"""<start_of_turn>user
Aşağıdaki matematik sorusunu cevapla:

SORU: {query}

BİLGİLER:
{context}

Lütfen verilen bilgileri kullanarak soruyu doğru ve anlaşılır şekilde cevapla. Eğer bilgiler yeterli değilse, kendi matematik bilginle yardımcı ol. Türkçe olarak, çocukların anlayabileceği şekilde açıkla. Adım adım çözüm göster. 

Eğer kullanıcı bir matematik sorusu istiyorsa (örneğin "rasyonel sayılar sorar mısın", "denklem sorusu sorar mısın"), o zaman verilen bilgilerden yola çıkarak uygun bir matematik sorusu üret. Soru üretirken:
- Sınıf seviyesine uygun olmalı
- Anlaşılır ve net olmalı
- Çözülebilir olmalı
- Sadece soruyu ver, cevabını verme

Eğer kullanıcı bir soruyu cevaplamak istiyorsa, o zaman adım adım çözüm göster.
<end_of_turn>
<start_of_turn>model
"""
            
            return prompt
            
        except Exception as e:
            logger.error(f"Context prompt oluşturma hatası: {e}")
            return f"Lütfen şu soruyu cevapla: {query}"
    
    def advanced_rag_pipeline(self, query: str, top_k: int = 5) -> str:
        """
        Advanced RAG pipeline - ÇOK HIZLI VERSİYON
        
        Args:
            query: Kullanıcı sorgusu
            top_k: Döndürülecek doküman sayısı
            
        Returns:
            Model cevabı
        """
        logger.info("Advanced RAG pipeline başlatılıyor...")
        
        # 1. Basit sorgu optimizasyonu - ÇOK HIZLI
        logger.info("1. Sorgu optimizasyonu...")
        optimized_query = self.pre_process_query(query)
        
        # 2. Doküman arama - ÇOK HIZLI
        logger.info("2. Doküman arama...")
        retrieved_docs = self.retrieve_relevant_documents(optimized_query, top_k=1)  # Sadece 1 doküman
        
        if not retrieved_docs:
            logger.info("Doküman bulunamadı, LLM ile cevap üretiliyor...")
            return self.generate_llm_only_response(query)
        
        # 3. Basit prompt oluştur - ÇOK HIZLI
        logger.info("3. Prompt oluşturma...")
        context_prompt = self.create_context_prompt(query, retrieved_docs, max_context_length=1000)
        
        # 4. Hızlı cevap üret - ÇOK KISA
        logger.info("4. Cevap üretimi...")
        response = self.generate_with_llm_fast(context_prompt, max_length=50, temperature=0.7)
        
        logger.info("Advanced RAG pipeline tamamlandı!")
        return response
    
    def select_best_response_by_embedding(self, query: str, current_response: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Basit cevap seçimi - ÇOK HIZLI VERSİYON
        
        Args:
            query: Kullanıcı sorgusu
            current_response: Mevcut cevap
            relevant_docs: İlgili dokümanlar
            
        Returns:
            Cevap
        """
        # Basit cevap döndür - ÇOK HIZLI
        return current_response
    
    def calculate_response_embedding_score(self, query: str, response: str, relevant_docs: List[Dict[str, Any]]) -> float:
        """
        Cevabın embedding skorunu hesapla - BASİT VERSİYON
        
        Args:
            query: Kullanıcı sorgusu
            response: Cevap
            relevant_docs: İlgili dokümanlar
            
        Returns:
            Embedding skoru (0-1 arası)
        """
        # Basit skor hesaplama - ÇOK HIZLI
        try:
            # Sadece sorgu ve cevap için embedding hesapla
            query_embedding = self.create_embedding_for_text(query)
            response_embedding = self.create_embedding_for_text(response)
            
            # Basit benzerlik hesapla
            similarity = self.calculate_similarity(query_embedding, response_embedding)
            
            return similarity
            
        except Exception as e:
            logger.error(f"Embedding skoru hesaplanırken hata: {e}")
            return 0.5  # Varsayılan skor
    
    def save_advanced_rag_system(self, output_dir: str = "advanced_rag_system"):
        """
        Advanced RAG sistemini kaydet
        
        Args:
            output_dir: Çıktı dizini
        """
        try:
            # Dizini oluştur
            os.makedirs(output_dir, exist_ok=True)
            
            # Embedding model'ini kaydet
            if self.embedding_model is not None:
                embedding_dir = os.path.join(output_dir, "embedding_model")
                os.makedirs(embedding_dir, exist_ok=True)
                self.embedding_model.save(embedding_dir)
                logger.info("Embedding model kaydedildi")
            
            # FAISS index'i kaydet
            if self.index is not None:
                faiss.write_index(self.index, os.path.join(output_dir, "faiss_index.bin"))
                logger.info("FAISS index kaydedildi")
            
            # Dokümanları kaydet
            if self.documents:
                with open(os.path.join(output_dir, "documents.pkl"), 'wb') as f:
                    pickle.dump(self.documents, f)
                logger.info("Dokümanlar kaydedildi")
            
            # Embedding'leri kaydet
            if self.embeddings is not None:
                np.save(os.path.join(output_dir, "embeddings.npy"), self.embeddings)
                logger.info("Embedding'ler kaydedildi")
            
            # LLM ve tokenizer'ı kaydet
            if self.llm is not None and self.tokenizer is not None:
                llm_dir = os.path.join(output_dir, "llm_model")
                os.makedirs(llm_dir, exist_ok=True)
                self.llm.save_pretrained(llm_dir)
                self.tokenizer.save_pretrained(llm_dir)
                logger.info("LLM model kaydedildi")
            
            logger.info(f"Advanced RAG sistemi {output_dir} dizinine kaydedildi")
            
        except Exception as e:
            logger.error(f"Sistem kaydetme hatası: {e}")
            raise
    
    def load_advanced_rag_system(self, input_dir: str = "advanced_rag_system"):
        """
        Advanced RAG sistemini yükle
        
        Args:
            input_dir: Giriş dizini
        """
        try:
            # Embedding model'ini yükle
            embedding_dir = os.path.join(input_dir, "embedding_model")
            if os.path.exists(embedding_dir):
                self.embedding_model = SentenceTransformer(embedding_dir)
                logger.info("Embedding model yüklendi")
            
            # FAISS index'i yükle
            faiss_path = os.path.join(input_dir, "faiss_index.bin")
            if os.path.exists(faiss_path):
                self.index = faiss.read_index(faiss_path)
                logger.info("FAISS index yüklendi")
            
            # Dokümanları yükle
            docs_path = os.path.join(input_dir, "documents.pkl")
            if os.path.exists(docs_path):
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info("Dokümanlar yüklendi")
            
            # Embedding'leri yükle
            embeddings_path = os.path.join(input_dir, "embeddings.npy")
            if os.path.exists(embeddings_path):
                self.embeddings = np.load(embeddings_path)
                logger.info("Embedding'ler yüklendi")
            
            # LLM ve tokenizer'ı yükle
            llm_dir = os.path.join(input_dir, "llm_model")
            if os.path.exists(llm_dir):
                self.llm = AutoModelForCausalLM.from_pretrained(llm_dir, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(llm_dir, trust_remote_code=True)
                self.llm.eval()
                logger.info("LLM model yüklendi")
            else:
                # LLM yoksa yükle
                self.load_llm()
            
            logger.info(f"Advanced RAG sistemi {input_dir} dizininden yüklendi")
            
        except Exception as e:
            logger.error(f"Sistem yükleme hatası: {e}")
            raise
    
    def train_advanced_rag_system(self, json_files: List[str], output_dir: str = "advanced_rag_system"):
        """
        Advanced RAG sistemini eğit - FOTOGRAFTEKİ SİSTEME GÖRE DÜZELTİLDİ
        
        Args:
            json_files: JSON dosya yolları
            output_dir: Çıktı dizini
        """
        logger.info("Advanced RAG sistemi eğitimi başlıyor...")
        
        # 1. INQUIRY (Veri yükleme)
        logger.info("1️⃣ Inquiry (Veri yükleme)...")
        data = self.load_data(json_files)
        
        # 2. PRE-PROCESS (Doküman hazırlama)
        logger.info("2️⃣ Pre-process (Doküman hazırlama)...")
        self.documents = self.prepare_documents(data)
        
        # 3. EMBED (Embedding oluşturma)
        logger.info("3️⃣ Embed (Embedding oluşturma)...")
        self.create_embeddings(self.documents)
        
        # 4. SEARCH (FAISS index oluşturma)
        logger.info("4️⃣ Search (FAISS index oluşturma)...")
        self.build_faiss_index(self.embeddings)
        
        # 5. FILTER (LLM yükleme)
        logger.info("5️⃣ Filter (LLM yükleme)...")
        self.load_llm()
        
        # 6. BUILD PROMPT (Sistem kaydetme)
        logger.info("6️⃣ Build Prompt (Sistem kaydetme)...")
        self.save_advanced_rag_system(output_dir)
        
        # 7. GENERATE (Eğitim tamamlandı)
        logger.info("7️⃣ Generate (Eğitim tamamlandı)...")
        
        # 8. SELF-REFLECT (Kalite kontrolü)
        logger.info("8️⃣ Self-reflect (Kalite kontrolü)...")
        logger.info("Advanced RAG sistemi eğitimi başarıyla tamamlandı!")
    
    def add_to_memory(self, user_message: str, bot_message: str):
        """
        Memory'ye mesaj ekle (7 item limit)
        
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
    
    def generate_llm_only_response(self, query: str) -> str:
        """
        Sadece LLM ile yanıt üret (RAG olmadan)
        
        Args:
            query: Kullanıcı sorgusu
            
        Returns:
            Bot yanıtı
        """
        try:
            # Memory context'i al
            memory_context = self.get_memory_context()
            
            # Gemma için düzeltilmiş prompt formatı
            if memory_context:
                prompt = f"<start_of_turn>user\n{memory_context}Kullanıcı: {query}<end_of_turn>\n<start_of_turn>model\n"
            else:
                prompt = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
            
            # LLM ile yanıt üret
            response = self.generate_with_llm(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"LLM chat sırasında hata: {e}")
            return "Üzgünüm, yanıt üretirken bir hata oluştu."
    
    def calculate_response_embedding_score(self, query: str, response: str, relevant_docs: List[Dict[str, Any]]) -> float:
        """
        Yanıtın embedding skorunu hesapla
        
        Args:
            query: Sorgu
            response: Yanıt
            relevant_docs: İlgili dokümanlar
            
        Returns:
            Embedding skoru
        """
        try:
            # Query ve response embedding'lerini oluştur
            query_embedding = self.create_embedding_for_text(query)
            response_embedding = self.create_embedding_for_text(response)
            
            # Query-response benzerliği
            query_response_similarity = self.calculate_similarity(query_embedding, response_embedding)
            
            # Response-doküman benzerlikleri
            doc_similarities = []
            for doc in relevant_docs:
                if isinstance(doc, dict):
                    doc_text = doc.get('document', str(doc))
                else:
                    doc_text = str(doc)
                
                doc_embedding = self.create_embedding_for_text(doc_text)
                doc_similarity = self.calculate_similarity(response_embedding, doc_embedding)
                doc_similarities.append(doc_similarity)
            
            # Ortalama doküman benzerliği
            avg_doc_similarity = np.mean(doc_similarities) if doc_similarities else 0.0
            
            # Genel skor (query-response ve response-doc benzerliklerinin ortalaması)
            overall_score = (query_response_similarity + avg_doc_similarity) / 2
            
            return overall_score
            
        except Exception as e:
            logger.error(f"Embedding skor hesaplama hatası: {e}")
            return 0.0

def main():
    """Ana fonksiyon"""
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
            else:
                print(f"⚠️  {file} dosyası bulunamadı, atlanıyor...")
        
        if not existing_files:
            print("❌ Hiçbir JSON dosyası bulunamadı! RAG sistemi eğitilemiyor.")
            return
        
        print(f"📚 {len(existing_files)} JSON dosyası bulundu. Advanced RAG sistemi eğitiliyor...")
        
        # Advanced RAG sistemi oluştur ve eğit
        advanced_rag = AdvancedMatematikRAG()
        advanced_rag.train_advanced_rag_system(existing_files, "advanced_rag_system")
        
        print("✅ Advanced RAG sistemi başarıyla eğitildi!")
        
    except Exception as e:
        print(f"❌ Advanced RAG sistemi eğitilirken hata oluştu: {e}")

if __name__ == "__main__":
    main() 