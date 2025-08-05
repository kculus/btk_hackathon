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

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedMatematikRAG:
    def __init__(self, model_path: str = "gemma-2-9b-it-tr-new"):
        """
        Advanced Matematik RAG sistemi
        
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
        self.max_embedding_score = 0.0  # En yüksek embedding skoru
        self.best_response = ""  # En iyi cevap
        
    def load_llm(self):
        """
        LLM model'ini yükle - CUDA OUT OF MEMORY SORUNU ÇÖZÜLMÜŞ VERSİYON
        """
        try:
            logger.info("LLM model yükleniyor...")
            
            # Tokenizer'ı yükle
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            
            if torch.cuda.is_available():
                # GPU optimizasyonları - Çok daha agresif memory yönetimi
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.5)  # Çok daha az memory kullan
                torch.cuda.memory.empty_cache()
                
                # Model yükleme ayarları - RTX 4090 için optimize edilmiş
                self.llm = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    max_memory={0: "4GB"},  # Çok daha az memory kullan
                    use_safetensors=True,
                    load_in_8bit=False,
                    load_in_4bit=False
                )
                
                # Model'i evaluation moduna al
                self.llm.eval()
                
                # Memory optimizasyonu
                torch.cuda.empty_cache()
                
            else:
                # CPU için normal yükleme
                self.llm = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=True,
                    use_safetensors=True
                )
            
            logger.info("LLM model başarıyla yüklendi")
            
        except Exception as e:
            logger.error(f"LLM model yüklenirken hata: {e}")
            # Hata durumunda alternatif yükleme dene - CPU'ya offload
            try:
                logger.info("Alternatif model yükleme yöntemi deneniyor (CPU offload)...")
                self.llm = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map={"": "cpu"},  # CPU'ya yükle
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                )
                self.llm.eval()
                logger.info("LLM model CPU'ya yüklendi")
            except Exception as e2:
                logger.error(f"CPU yükleme de başarısız: {e2}")
                raise
    
    def generate_with_llm(self, prompt: str, max_length: int = 256, temperature: float = 0.7) -> str:
        """
        LLM ile metin üret - DEVICE MİSMATCH SORUNU ÇÖZÜLMÜŞ VERSİYON
        
        Args:
            prompt: Input prompt
            max_length: Maksimum token uzunluğu
            temperature: Sıcaklık parametresi
            
        Returns:
            Üretilen metin
        """
        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            # Device kontrolü ve düzeltme
            device = "cuda" if torch.cuda.is_available() and hasattr(self.llm, 'device') and self.llm.device.type == 'cuda' else "cpu"
            
            # Input'ları doğru device'a taşı
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Model'i doğru device'da tut
            if device == "cuda":
                if not hasattr(self.llm, 'device') or self.llm.device.type != 'cuda':
                    self.llm = self.llm.cuda()
                # Memory temizle
                torch.cuda.empty_cache()
            else:
                # CPU'da çalıştır
                self.llm = self.llm.cpu()
            
            # Generate with optimized settings for Gemma model
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.encode("<end_of_turn>")[0] if "<end_of_turn>" in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    num_return_sequences=1
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Prompt'u çıkar
            response = response[len(prompt):].strip()
            
            # Memory temizle
            if device == "cuda":
                torch.cuda.empty_cache()
            
            return response
            
        except Exception as e:
            logger.error(f"LLM üretiminde hata: {e}")
            return ""
    
    def create_embedding_for_text(self, text: str) -> np.ndarray:
        """
        Metin için embedding oluştur
        
        Args:
            text: Embedding oluşturulacak metin
            
        Returns:
            Embedding vektörü
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model yüklenmemiş")
        
        embedding = self.embedding_model.encode([text])
        return embedding[0]
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        İki embedding arasındaki benzerliği hesapla
        
        Args:
            embedding1: İlk embedding
            embedding2: İkinci embedding
            
        Returns:
            Benzerlik skoru (0-1 arası)
        """
        # Cosine similarity hesapla
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return float(similarity)
    
    def pre_process_query(self, query: str) -> str:
        """
        Sorguyu optimize et (Advanced RAG - Pre-process adımı) - HIZLANDIRILMIŞ VERSİYON
        
        Args:
            query: Orijinal sorgu
            
        Returns:
            Optimize edilmiş sorgu
        """
        # Basit ve hızlı optimizasyon - LLM çağırmadan
        query = query.lower().strip()
        
        # Gereksiz kelimeleri kaldır
        stop_words = ['lütfen', 'acaba', 'şu', 'bu', 'o', 'bir', 'birkaç', 'bazı', 'her', 'tüm', 'hiç', 'çok', 'az', 'daha', 'en', 'pek', 'gayet', 'oldukça', 'biraz', 'kadar', 'gibi', 'kadar', 'için', 'ile', 've', 'veya', 'ama', 'fakat', 'ancak', 'lakin', 'ne', 'nasıl', 'nerede', 'ne zaman', 'kim', 'hangi', 'kaç', 'neden', 'niçin', 'niye']
        
        # Matematik terimlerini koru
        math_terms = ['toplama', 'çıkarma', 'çarpma', 'bölme', 'kesir', 'ondalık', 'yüzde', 'oran', 'orantı', 'denklem', 'eşitlik', 'geometri', 'alan', 'çevre', 'hacim', 'açı', 'üçgen', 'dikdörtgen', 'kare', 'daire', 'çember', 'küp', 'küre', 'silindir', 'prizma', 'piramit', 'sayı', 'rakam', 'basamak', 'bölüm', 'kalan', 'çarpan', 'kat', 'asal', 'bölünebilme', 'üs', 'kök', 'logaritma', 'trigonometri', 'sinüs', 'kosinüs', 'tanjant', 'küp', 'kare', 'kök', 'mutlak', 'değer', 'modül', 'faktöriyel', 'kombinasyon', 'permütasyon', 'olasılık', 'istatistik', 'ortalama', 'medyan', 'mod', 'varyans', 'standart', 'sapma', 'grafik', 'tablo', 'veri', 'analiz']
        
        # Kelimeleri ayır
        words = query.split()
        
        # Önemli kelimeleri koru
        important_words = []
        for word in words:
            # Matematik terimlerini her zaman koru
            if any(math_term in word for math_term in math_terms):
                important_words.append(word)
            # Sayıları koru
            elif any(char.isdigit() for char in word):
                important_words.append(word)
            # Stop word değilse koru
            elif word not in stop_words and len(word) > 2:
                important_words.append(word)
        
        # Optimize edilmiş sorguyu oluştur
        optimized_query = ' '.join(important_words)
        
        # Eğer çok kısa olduysa orijinal sorguyu döndür
        if len(optimized_query.strip()) < 3:
            return query
        
        return optimized_query.strip()
    
    def filter_relevant_documents(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        İlgili dokümanları filtrele (Advanced RAG - Filter adımı) - HIZLANDIRILMIŞ VERSİYON
        
        Args:
            query: Kullanıcı sorgusu
            retrieved_docs: Getirilen dokümanlar
            
        Returns:
            Filtrelenmiş dokümanlar
        """
        if not retrieved_docs:
            return []
        
        # Basit keyword-based filtreleme - LLM çağırmadan
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Matematik terimlerini tanımla
        math_terms = ['toplama', 'çıkarma', 'çarpma', 'bölme', 'kesir', 'ondalık', 'yüzde', 'oran', 'orantı', 'denklem', 'eşitlik', 'geometri', 'alan', 'çevre', 'hacim', 'açı', 'üçgen', 'dikdörtgen', 'kare', 'daire', 'çember', 'küp', 'küre', 'silindir', 'prizma', 'piramit', 'sayı', 'rakam', 'basamak', 'bölüm', 'kalan', 'çarpan', 'kat', 'asal', 'bölünebilme', 'üs', 'kök', 'logaritma', 'trigonometri', 'sinüs', 'kosinüs', 'tanjant', 'küp', 'kare', 'kök', 'mutlak', 'değer', 'modül', 'faktöriyel', 'kombinasyon', 'permütasyon', 'olasılık', 'istatistik', 'ortalama', 'medyan', 'mod', 'varyans', 'standart', 'sapma', 'grafik', 'tablo', 'veri', 'analiz']
        
        # Sorgudaki matematik terimlerini bul
        query_math_terms = [term for term in math_terms if term in query_lower]
        
        filtered_docs = []
        for doc in retrieved_docs:
            doc_text_lower = doc['document'].lower()
            score = 0
            
            # Matematik terimlerinin eşleşmesini kontrol et
            for math_term in query_math_terms:
                if math_term in doc_text_lower:
                    score += 2  # Matematik terimleri daha önemli
            
            # Genel kelime eşleşmesini kontrol et
            for word in query_words:
                if word in doc_text_lower and len(word) > 2:
                    score += 1
            
            # Sayı eşleşmesini kontrol et
            query_numbers = re.findall(r'\d+', query)
            doc_numbers = re.findall(r'\d+', doc_text_lower)
            for num in query_numbers:
                if num in doc_numbers:
                    score += 1
            
            # Skor yeterliyse dokümanı ekle
            if score > 0:
                filtered_docs.append((doc, score))
        
        # Skora göre sırala ve en iyilerini al
        filtered_docs.sort(key=lambda x: x[1], reverse=True)
        
        # En iyi 3 dokümanı döndür (veya hepsini eğer 3'ten azsa)
        result = [doc for doc, score in filtered_docs[:3]]
        
        # Eğer filtreleme sonucu boşsa, tüm dokümanları döndür
        if not result:
            return retrieved_docs
        
        return result
    
    def self_reflect_and_improve_with_embeddings(self, query: str, initial_response: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Cevabı embedding'lerle birlikte değerlendir ve iyileştir (Advanced RAG - Self-reflect adımı)
        
        Args:
            query: Kullanıcı sorgusu
            initial_response: İlk üretilen cevap
            relevant_docs: İlgili dokümanlar
            
        Returns:
            İyileştirilmiş cevap
        """
        try:
            # Sorgu ve cevap için embedding oluştur
            query_embedding = self.create_embedding_for_text(query)
            response_embedding = self.create_embedding_for_text(initial_response)
            
            # İlgili dokümanların embedding'lerini hesapla
            doc_embeddings = []
            for doc in relevant_docs:
                doc_embedding = self.create_embedding_for_text(doc['document'])
                doc_embeddings.append(doc_embedding)
            
            # Cevabın dokümanlarla benzerliğini hesapla
            doc_similarities = []
            for doc_embedding in doc_embeddings:
                similarity = self.calculate_similarity(response_embedding, doc_embedding)
                doc_similarities.append(similarity)
            
            # Ortalama benzerlik
            avg_similarity = np.mean(doc_similarities) if doc_similarities else 0.0
            
            # Sorgu-cevap benzerliği
            query_response_similarity = self.calculate_similarity(query_embedding, response_embedding)
            
            # Embedding tabanlı değerlendirme
            embedding_evaluation = f"""
Embedding Analizi:
- Cevap-Doküman Ortalama Benzerlik: {avg_similarity:.3f}
- Sorgu-Cevap Benzerlik: {query_response_similarity:.3f}
- Beklenen Minimum Benzerlik: 0.5
"""
            
            # LLM tabanlı değerlendirme
            prompt = f"""Aşağıdaki matematik sorusu ve cevabını değerlendir:

Soru: {query}

Cevap: {initial_response}

{embedding_evaluation}

Bu cevap doğru ve kapsamlı mı? Embedding analizi de dikkate alarak değerlendir.
Eğer değilse, daha iyi bir cevap yaz. Eğer doğruysa "DOĞRU" yaz:"""
            
            reflection = self.generate_with_llm(prompt, max_length=512, temperature=0.3)
            
            # Embedding skorlarına göre değerlendirme - daha esnek eşikler
            if avg_similarity < 0.5 or query_response_similarity < 0.4:
                logger.info(f"Embedding skorları düşük: Doc={avg_similarity:.3f}, Query={query_response_similarity:.3f}")
                # Cevabı iyileştir
                improvement_prompt = f"""Aşağıdaki cevabı daha iyi hale getir:

Soru: {query}
Mevcut Cevap: {initial_response}

Daha detaylı ve doğru bir cevap yaz:"""
                
                improved_response = self.generate_with_llm(improvement_prompt, max_length=512, temperature=0.3)
                
                if improved_response.strip():
                    return improved_response.strip()
            
            # Eğer "DOĞRU" içeriyorsa orijinal cevabı döndür
            if "DOĞRU" in reflection.upper():
                return initial_response
            
            # İyileştirilmiş cevabı döndür
            return reflection.strip()
            
        except Exception as e:
            logger.error(f"Self-reflect sırasında hata: {e}")
            # Hata durumunda basit değerlendirme yap
            return self.self_reflect_and_improve(query, initial_response)
    
    def self_reflect_and_improve(self, query: str, initial_response: str) -> str:
        """
        Cevabı değerlendir ve iyileştir (Basit versiyon)
        
        Args:
            query: Kullanıcı sorgusu
            initial_response: İlk üretilen cevap
            
        Returns:
            İyileştirilmiş cevap
        """
        prompt = f"""Aşağıdaki matematik sorusu ve cevabını değerlendir:

Soru: {query}

Cevap: {initial_response}

Bu cevap doğru ve kapsamlı mı? Eğer değilse, daha iyi bir cevap yaz. Eğer doğruysa "DOĞRU" yaz:"""
        
        reflection = self.generate_with_llm(prompt, max_length=512, temperature=0.3)
        
        # Eğer "DOĞRU" içeriyorsa orijinal cevabı döndür
        if "DOĞRU" in reflection.upper():
            return initial_response
        
        # İyileştirilmiş cevabı döndür
        return reflection.strip()
    
    def load_data(self, json_files: List[str]) -> List[Dict[str, Any]]:
        """
        JSON dosyalarından veri yükle
        
        Args:
            json_files: JSON dosya yolları listesi
            
        Returns:
            Yüklenen veri listesi
        """
        all_data = []
        
        for file_path in json_files:
            try:
                logger.info(f"{file_path} dosyası yükleniyor...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Her veri öğesine kaynak dosya bilgisi ekle
                    for item in data:
                        item['source'] = file_path
                    all_data.extend(data)
                logger.info(f"{file_path} dosyası yüklendi: {len(data)} öğe")
            except FileNotFoundError:
                logger.error(f"{file_path} dosyası bulunamadı!")
            except json.JSONDecodeError as e:
                logger.error(f"{file_path} dosyası JSON formatında değil: {e}")
            except Exception as e:
                logger.error(f"{file_path} dosyası yüklenirken hata: {e}")
                
        logger.info(f"Toplam {len(all_data)} öğe yüklendi")
        return all_data
    
    def prepare_documents(self, data: List[Dict[str, Any]]) -> List[str]:
        """
        Veriyi RAG için doküman formatına çevir - DÜZELTİLMİŞ VERSİYON
        
        Args:
            data: Yüklenen veri
            
        Returns:
            Doküman listesi
        """
        documents = []
        
        for item in data:
            # Daha iyi doküman formatı oluştur
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            output = item.get('output', '')
            
            # Matematik sorusu formatında doküman oluştur
            if instruction and input_text and output:
                doc = f"Soru: {instruction}\n"
                doc += f"Problem: {input_text}\n"
                doc += f"Cevap: {output}\n"
                
                # Ek bilgiler ekle
                if "toplama" in instruction.lower() or "+" in input_text:
                    doc += "Konu: Toplama işlemi\n"
                elif "çıkarma" in instruction.lower() or "-" in input_text:
                    doc += "Konu: Çıkarma işlemi\n"
                elif "çarpma" in instruction.lower() or "×" in input_text or "*" in input_text:
                    doc += "Konu: Çarpma işlemi\n"
                elif "bölme" in instruction.lower() or "÷" in input_text or "/" in input_text:
                    doc += "Konu: Bölme işlemi\n"
                elif "kesir" in instruction.lower():
                    doc += "Konu: Kesirler\n"
                elif "ondalık" in instruction.lower():
                    doc += "Konu: Ondalık sayılar\n"
                elif "geometri" in instruction.lower():
                    doc += "Konu: Geometri\n"
                elif "para" in instruction.lower() or "tl" in input_text.lower():
                    doc += "Konu: Para\n"
                elif "zaman" in instruction.lower():
                    doc += "Konu: Zaman ölçme\n"
                elif "alan" in instruction.lower():
                    doc += "Konu: Alan hesaplama\n"
                elif "çevre" in instruction.lower():
                    doc += "Konu: Çevre hesaplama\n"
                else:
                    doc += "Konu: Matematik\n"
                
                documents.append(doc)
            
        logger.info(f"Toplam {len(documents)} doküman hazırlandı")
        return documents
    
    def create_embeddings(self, documents: List[str], model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Dokümanlar için embedding oluştur
        
        Args:
            documents: Doküman listesi
            model_name: Embedding model adı
        """
        logger.info("Embedding model yükleniyor...")
        self.embedding_model = SentenceTransformer(model_name)
        
        logger.info("Embedding'ler oluşturuluyor...")
        self.embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        
        logger.info(f"Embedding'ler oluşturuldu: {self.embeddings.shape}")
        
    def build_faiss_index(self, embeddings: np.ndarray):
        """
        FAISS index oluştur
        
        Args:
            embeddings: Embedding'ler
        """
        logger.info("FAISS index oluşturuluyor...")
        
        # FAISS index oluştur
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        
        # Embedding'leri normalize et (cosine similarity için)
        faiss.normalize_L2(embeddings)
        
        # Index'e ekle
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index oluşturuldu: {self.index.ntotal} vektör")
        
    def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Sorgu için ilgili dokümanları getir - DÜZELTİLMİŞ VERSİYON
        
        Args:
            query: Sorgu metni
            top_k: Döndürülecek doküman sayısı
            
        Returns:
            İlgili dokümanlar listesi
        """
        if self.embedding_model is None or self.index is None:
            logger.error("RAG sistemi yüklenmemiş")
            return []
        
        try:
            # Sorgu embedding'i oluştur
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Benzerlik hesapla
            similarities, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Sonuçları hazırla
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.documents):  # Index kontrolü
                    results.append({
                        'document': self.documents[idx],
                        'similarity': float(similarity),
                        'rank': i + 1
                    })
            
            logger.info(f"Retrieved {len(results)} documents with similarities: {[r['similarity'] for r in results]}")
            return results
            
        except Exception as e:
            logger.error(f"Doküman arama sırasında hata: {e}")
            return []
    
    def create_context_prompt(self, query: str, relevant_docs: list, max_context_length: int = 2000) -> str:
        """
        Sorgu ve ilgili dokümanlardan context prompt oluştur
        
        Args:
            query: Kullanıcı sorgusu
            relevant_docs: İlgili dokümanlar
            max_context_length: Maksimum context uzunluğu
            
        Returns:
            Context prompt
        """
        # İlgili dokümanları birleştir
        context_parts = []
        current_length = 0
        
        for doc in relevant_docs:
            doc_text = doc['document']
            if current_length + len(doc_text) < max_context_length:
                context_parts.append(doc_text)
                current_length += len(doc_text)
            else:
                break
        
        context = "\n\n".join(context_parts)
        
        # Prompt template
        prompt = f"""Aşağıdaki matematik bilgilerini kullanarak soruyu yanıtla:

Bilgiler:
{context}

Soru: {query}

Cevap:"""
        
        return prompt
    
    def advanced_rag_pipeline(self, query: str, top_k: int = 5) -> str:
        """
        Advanced RAG pipeline'ı çalıştır - HIZLANDIRILMIŞ VERSİYON
        
        Args:
            query: Kullanıcı sorgusu
            top_k: Döndürülecek doküman sayısı
            
        Returns:
            Final cevap
        """
        # Embedding skorlarını sıfırla
        self.max_embedding_score = 0.0
        self.best_response = ""
        
        # 1. Pre-process (Sorgu optimizasyonu) - HIZLANDIRILMIŞ
        optimized_query = self.pre_process_query(query)
        
        # 2. Embed & Search
        retrieved_docs = self.retrieve_relevant_documents(optimized_query, top_k=top_k)
        
        if not retrieved_docs:
            return self.generate_llm_only_response(query)
        
        # 3. Filter (Doküman filtreleme) - HIZLANDIRILMIŞ
        filtered_docs = self.filter_relevant_documents(query, retrieved_docs)
        
        # 4. Build Prompt & Generate
        context_prompt = self.create_context_prompt(query, filtered_docs)
        initial_response = self.generate_with_llm(context_prompt, max_length=512, temperature=0.7)
        
        # 5. Self-reflect (Cevap değerlendirme ve iyileştirme) - Embedding'lerle
        final_response = self.self_reflect_and_improve_with_embeddings(query, initial_response, filtered_docs)
        
        # 6. Embedding skoruna göre en iyi cevabı seç
        best_response = self.select_best_response_by_embedding(query, final_response, filtered_docs)
        
        # 7. Embedding skoru kontrolü - 0.5'ten düşükse LLM ile cevap üret
        if self.max_embedding_score < 0.5:
            return self.generate_llm_only_response(query)
        
        return best_response
    
    def save_advanced_rag_system(self, output_dir: str = "advanced_rag_system"):
        """
        Advanced RAG sistemini kaydet
        
        Args:
            output_dir: Çıktı dizini
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Embedding model'i kaydet
        if self.embedding_model:
            logger.info("Embedding model kaydediliyor...")
            self.embedding_model.save(os.path.join(output_dir, "embedding_model"))
        
        # FAISS index'i kaydet
        if self.index:
            logger.info("FAISS index kaydediliyor...")
            faiss.write_index(self.index, os.path.join(output_dir, "faiss_index.bin"))
        
        # Dokümanları kaydet
        logger.info("Dokümanlar kaydediliyor...")
        with open(os.path.join(output_dir, "documents.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Embedding'leri kaydet
        if self.embeddings is not None:
            logger.info("Embedding'ler kaydediliyor...")
            np.save(os.path.join(output_dir, "embeddings.npy"), self.embeddings)
        
        logger.info(f"Advanced RAG sistemi {output_dir} dizinine kaydedildi")
        
    def load_advanced_rag_system(self, input_dir: str = "advanced_rag_system"):
        """
        Advanced RAG sistemini yükle
        
        Args:
            input_dir: Giriş dizini
        """
        # Embedding model'i yükle - eğer kaydedilmişse onu kullan, yoksa yeniden yükle
        embedding_model_path = os.path.join(input_dir, "embedding_model")
        if os.path.exists(embedding_model_path):
            logger.info("Kaydedilmiş embedding model yükleniyor...")
            self.embedding_model = SentenceTransformer(embedding_model_path)
        else:
            logger.info("Embedding model HuggingFace'den yükleniyor...")
            self.embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # FAISS index'i yükle
        index_path = os.path.join(input_dir, "faiss_index.bin")
        if os.path.exists(index_path):
            logger.info("FAISS index yükleniyor...")
            self.index = faiss.read_index(index_path)
        else:
            raise ValueError("FAISS index dosyası bulunamadı")
        
        # Dokümanları yükle
        documents_path = os.path.join(input_dir, "documents.pkl")
        if os.path.exists(documents_path):
            logger.info("Dokümanlar yükleniyor...")
            with open(documents_path, 'rb') as f:
                self.documents = pickle.load(f)
        else:
            raise ValueError("Dokümanlar dosyası bulunamadı")
        
        # Embedding'leri yükle
        embeddings_path = os.path.join(input_dir, "embeddings.npy")
        if os.path.exists(embeddings_path):
            logger.info("Embedding'ler yükleniyor...")
            self.embeddings = np.load(embeddings_path)
        else:
            raise ValueError("Embedding'ler dosyası bulunamadı")
        
        logger.info("Advanced RAG sistemi başarıyla yüklendi")
    
    def train_advanced_rag_system(self, json_files: List[str], output_dir: str = "advanced_rag_system"):
        """
        Advanced RAG sistemini eğit
        
        Args:
            json_files: JSON dosya yolları
            output_dir: Çıktı dizini
        """
        logger.info("Advanced RAG sistemi eğitimi başlıyor...")
        
        # Önce veri yükle
        logger.info("1. Veri yükleme...")
        data = self.load_data(json_files)
        
        # Dokümanları hazırla
        logger.info("2. Doküman hazırlama...")
        self.documents = self.prepare_documents(data)
        
        # Embedding'leri oluştur
        logger.info("3. Embedding oluşturma...")
        self.create_embeddings(self.documents)
        
        # FAISS index oluştur
        logger.info("4. FAISS index oluşturma...")
        self.build_faiss_index(self.embeddings)
        
        # LLM yükle (en son)
        logger.info("5. LLM yükleme...")
        self.load_llm()
        
        # Sistemi kaydet
        logger.info("6. Sistem kaydetme...")
        self.save_advanced_rag_system(output_dir)
        
        logger.info("Advanced RAG sistemi eğitimi tamamlandı!")

    def select_best_response_by_embedding(self, query: str, current_response: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Embedding skoruna göre en iyi cevabı seç
        
        Args:
            query: Kullanıcı sorgusu
            current_response: Mevcut cevap
            relevant_docs: İlgili dokümanlar
            
        Returns:
            En iyi cevap
        """
        try:
            # Mevcut cevabın embedding skorunu hesapla
            current_score = self.calculate_response_embedding_score(query, current_response, relevant_docs)
            
            # Eğer mevcut cevap 0.7'den yüksekse direkt döndür
            if current_score >= 0.7:
                logger.info(f"Mevcut cevap yeterince iyi (skor: {current_score:.3f})")
                self.max_embedding_score = current_score
                self.best_response = current_response
                return current_response
            
            # Eğer mevcut cevap 0.5'ten düşükse LLM ile cevap üret
            if current_score < 0.5:
                logger.info(f"Mevcut cevap skoru düşük (skor: {current_score:.3f} < 0.5), LLM ile cevap üretiliyor...")
                llm_response = self.generate_llm_only_response(query)
                self.max_embedding_score = 0.0  # LLM cevabı için embedding skoru 0
                self.best_response = llm_response
                return llm_response
            
            # Mevcut cevabı en iyi olarak kaydet
            self.max_embedding_score = current_score
            self.best_response = current_response
            
            # Dokümanlardan alternatif cevaplar üret ve karşılaştır
            logger.info("Alternatif cevaplar üretiliyor...")
            
            for i, doc in enumerate(relevant_docs[:3]):  # İlk 3 dokümanı dene
                try:
                    # Doküman içeriğinden cevap üret
                    doc_prompt = f"""Aşağıdaki matematik bilgisini kullanarak soruyu yanıtla:

Bilgi: {doc['document'][:500]}

Soru: {query}

Cevap:"""
                    
                    alt_response = self.generate_with_llm(doc_prompt, max_length=256, temperature=0.5)
                    
                    if alt_response.strip():
                        # Alternatif cevabın embedding skorunu hesapla
                        alt_score = self.calculate_response_embedding_score(query, alt_response, relevant_docs)
                        
                        logger.info(f"Alternatif {i+1} skor: {alt_score:.3f}")
                        
                        # Eğer daha yüksek skor varsa güncelle
                        if alt_score > self.max_embedding_score:
                            self.max_embedding_score = alt_score
                            self.best_response = alt_response
                            logger.info(f"Yeni en iyi cevap bulundu (skor: {alt_score:.3f})")
                        
                        # Eğer 0.7'yi geçtiyse dur
                        if alt_score >= 0.7:
                            logger.info(f"Yeterince iyi cevap bulundu (skor: {alt_score:.3f})")
                            break
                            
                except Exception as e:
                    logger.error(f"Alternatif cevap üretirken hata: {e}")
                    continue
            
            return self.best_response
            
        except Exception as e:
            logger.error(f"En iyi cevap seçerken hata: {e}")
            return current_response
    
    def generate_llm_only_response(self, query: str) -> str:
        """
        Sadece LLM ile cevap üret (RAG olmadan)
        
        Args:
            query: Kullanıcı sorgusu
            
        Returns:
            LLM ile üretilen cevap
        """
        try:
            # Matematik odaklı prompt
            prompt = f"""Sen matematik öğretmeni bir tavşansın. Aşağıdaki soruyu çocuk dostu bir şekilde yanıtla:

Soru: {query}

Cevap:"""
            
            response = self.generate_with_llm(prompt, max_length=512, temperature=0.7)
            
            if not response.strip():
                # Alternatif prompt
                alt_prompt = f"""Aşağıdaki matematik sorusunu yanıtla:

{query}

Cevap:"""
                response = self.generate_with_llm(alt_prompt, max_length=512, temperature=0.7)
            
            return response.strip() if response.strip() else "Üzgünüm, bu soruya cevap veremiyorum."
            
        except Exception as e:
            logger.error(f"LLM ile cevap üretirken hata: {e}")
            return "Üzgünüm, yanıt üretirken bir hata oluştu."
    
    def calculate_response_embedding_score(self, query: str, response: str, relevant_docs: List[Dict[str, Any]]) -> float:
        """
        Cevabın embedding skorunu hesapla - DÜZELTİLMİŞ VERSİYON
        
        Args:
            query: Kullanıcı sorgusu
            response: Cevap
            relevant_docs: İlgili dokümanlar
            
        Returns:
            Embedding skoru (0-1 arası)
        """
        try:
            if not relevant_docs:
                return 0.0
            
            # Sorgu ve cevap için embedding oluştur
            query_embedding = self.create_embedding_for_text(query)
            response_embedding = self.create_embedding_for_text(response)
            
            # İlgili dokümanların embedding'lerini hesapla
            doc_similarities = []
            for doc in relevant_docs:
                try:
                    doc_embedding = self.create_embedding_for_text(doc['document'])
                    similarity = self.calculate_similarity(response_embedding, doc_embedding)
                    doc_similarities.append(similarity)
                except Exception as e:
                    logger.error(f"Doküman embedding hesaplanırken hata: {e}")
                    continue
            
            # Ortalama doküman benzerliği
            avg_doc_similarity = np.mean(doc_similarities) if doc_similarities else 0.0
            
            # Sorgu-cevap benzerliği
            query_response_similarity = self.calculate_similarity(query_embedding, response_embedding)
            
            # Toplam skor (ağırlıklı ortalama)
            total_score = (avg_doc_similarity * 0.6) + (query_response_similarity * 0.4)
            
            logger.info(f"Embedding skorları: Doc={avg_doc_similarity:.3f}, Query={query_response_similarity:.3f}, Total={total_score:.3f}")
            
            return total_score
            
        except Exception as e:
            logger.error(f"Embedding skoru hesaplanırken hata: {e}")
            return 0.0

def main():
    """Ana fonksiyon"""
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
    
    # Advanced RAG sistemi oluştur
    advanced_rag = AdvancedMatematikRAG()
    
    # Sistemi eğit
    advanced_rag.train_advanced_rag_system(json_files)
    
    # Test sorgusu
    test_query = "Toplama işlemi nasıl yapılır?"
    response = advanced_rag.advanced_rag_pipeline(test_query)
    
    print(f"\nSorgu: {test_query}")
    print(f"Cevap: {response}")

if __name__ == "__main__":
    main() 