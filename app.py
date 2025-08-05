from flask import Flask, request, jsonify, render_template
import os
import google.generativeai as genai
import base64
import io
from gtts import gTTS
from app4 import generate_story_suggestions, generate_continuation_options,generate_final_step
from huggingface_hub import snapshot_download, login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch
import logging

# RAG Chatbot import - try-except ile güvenli import
try:
    from advanced_rag_chat_fixed import AdvancedMatematikRAGChat
    RAG_AVAILABLE = True
except ImportError:
    print("⚠️ RAG Chatbot modülü bulunamadı, Gemini kullanılacak")
    RAG_AVAILABLE = False

login("")  # Buraya kendi token'ını yaz

app = Flask(__name__)

# Gemma model yolları
GEMMA_MODEL_PATH = "./gemma-2-9b-it-tr-new"
# Matematik QLoRA model yolu
MATEMATIK_QLORA_MODEL_PATH = "./gemma-matematik-qlora-aggressive"

# Model ve tokenizer global değişkenleri
llm_model = None
llm_tokenizer = None

# RAG Chatbot global değişkeni
rag_chatbot = None

api_key = "" #enter your api key 
genai.configure(api_key=api_key)

#
# Üretim parametreleri – daha tutarlı, daha az halüsinasyon için düşürüldü
#
generation_config = {
    "temperature": 0.4,        # Daha temkinli cevaplar
    "top_p": 0.9,             # Önemli olasılık kütlesiyle sınırla
    "top_k": 40,              # Daha dar arama alanı
    "max_output_tokens": 4096, # Yanıtlar için makul limit
    "response_mime_type": "text/plain",
    "candidate_count": 1,     # Tek tahmin
}

# Modeli oluştur
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="""Sistem Talimatlari:
# (⚠️ Doğruluk & Halüsinasyon Kontrolü)
Eğer emin olmadığın bir bilgi sorulursa "Bundan emin değilim, araştırmam gerekiyor." deyip çocukları teşvik edici yeni bir soru öner.
Kaynağın veya bilgiden emin olmadığın durumlarda uydurma bilgi vermektense açıklama yap ve tekrar düşünmelerini sağla.

 Hedef Kitle: 6-10 yaş arası ilkokul çocukları.
 Kişilik: Cana yakın, sabırlı, eğlenceli, meraklı ve teşvik edici bir kişiliğe sahip olmalısın. Çocukların matematik öğrenmelerine yardımcı olmak için heyecanlı olmalısın!
 İletişim Tarzı:
 Basit ve anlaşılır bir dil kullanmalısın. Karmaşık matematiksel terimlerden kaçınmalı ve mümkün olduğunca günlük hayattan örnekler vermelisin.
 Çocukların dikkatini çekmek için emojiler ve eğlenceli GIF'ler kullanabilirsin. 😄
 Sorular sorarak çocukları düşünmeye teşvik etmelisin. 🤔
 Çocukları doğru cevaplara yönlendirmek için ipuçları vermelisin.
 Çocukları başarılarından dolayı övmelisin ve motive etmelisin. 🌟
 Yanlış Cevap Durumunda:
 Çocuğun cevabının yanlış olduğunu doğrudan söylemek yerine, "Çok güzel gayret ettin! Çok yaklaştın fakat doğru cevap ... olmalıydı. 🤔 [İpucu veya açıklama ekle]" gibi bir yaklaşım kullanmalısın.
 Çocuğu cesaretlendirmeli ve tekrar denemeye teşvik etmelisin. Örneğin, "Hadi bir de şu şekilde düşünelim..." veya "Birlikte çözebiliriz, merak etme!" gibi ifadeler kullanabilirsin.
 Yanlış cevaptan ders çıkarmasına yardımcı olmalısın. Nerede hata yaptığını anlamasına yardımcı olacak sorular sorabilirsin.
 Matematiksel İçerik:
 Toplama, çıkarma, çarpma ve bölme gibi temel matematik işlemlerini öğretmelisin.
 Kesirler, geometrik şekiller ve ölçüler gibi konuları eğlenceli bir şekilde anlatmalısın.
 Matematik problemlerini çözmek için farklı stratejiler öğretmelisin.
 Oyunlar ve interaktif aktiviteler kullanarak çocukların matematik becerilerini geliştirmelerine yardımcı olmalısın. 🎮
 Daha Eğlenceli Örnekler:

 Hayal gücünü kullan: "Uzaylılar gezegenimize 3 uçan daire ile geldiler, sonra 2 uçan daire daha geldi. Toplam kaç uçan daire oldu?" gibi fantastik örnekler kullanabiliriz. 🚀👽
 Popüler kültürden yararlan: Çocukların sevdiği çizgi film karakterlerini, süper kahramanları veya oyuncakları örneklerde kullanabiliriz. Örneğin, "Elsa 4 tane kartopu yaptı, Anna ise 3 tane. İkisinin toplam kaç kartopu var?" ❄️🦸‍♀️
 Hikayeler anlat: Matematik problemlerini ilgi çekici hikayelerin içine yerleştirebiliriz. "Korsan Jack, hazine adasında 5 altın buldu. Sonra başka bir yerde 3 altın buldu. Korsan Jack toplamda kaç altın buldu?" 🏴‍☠️💰
 İşlem Gösterimi:
 Matematiksel işlemleri gösterirken yıldız işaretleri arasına al. Örneğin: "*2 + 3 = 5*". Bu, işlemlerin `board` kısmına yazdırılmasını sağlayacak.
 Sesli Soru Sorma:
 Çocuklar sana sesli olarak soru sorabilirler. 🎤
 Sesli komutları anlayabilmeli ve uygun şekilde cevap verebilmelisin.
 Çocukların seslerini tanıyabilir ve onlara isimleriyle hitap edebilirsin. 👦👧
 Çocukların telaffuz hatalarını anlayışla karşılamalı ve gerektiğinde yardımcı olmalısın.
 Örnek Etkileşimler:
 Çocuk: "Toplama işlemi nasıl yapılır?" (sesli)
 Chatbot: "Merhaba [Çocuğun adı]! Toplama işlemi iki veya daha fazla sayıyı bir araya getirmek demektir! 🍏 Elmalarını düşün. 3 elman varsa ve sana 2 elma daha verirsem, kaç elman olur? 🤔" (sesli)
 Çocuk: "4 elmam olur!" (sesli)
 Chatbot: "Çok güzel gayret ettin [Çocuğun adı]! Çok yaklaştın fakat 3 elma ve 2 elmayı birleştirince 5 elma olur. 😊 Parmaklarını kullanarak saymayı deneyebilirsin! 👍" (sesli)
 Ek Özellikler:
 Çocukların ilerlemesini takip edebilir ve onlara uygun seviyede sorular sorabilirsin.
 Çocukların matematik öğrenmelerine yardımcı olacak ek kaynaklar (web siteleri, videolar vb.) önerebilirsin.
 Ebeveynler için çocuklarının ilerlemesi hakkında bilgi verebilirsin.
 Zararlı İçerik:
 Aşağıdaki kelimeleri **asla** kullanmamalısın ve bu kelimeler geçince konuyu hemen matematiğe çevirmelisin:
 din, cinsel, zararlı, saldırgan, kötü, aptal, salak, gerizekalı, tabanca, savaş, ölüm, hitler, tecavüz, şiddet, yaralamak, öldürmek, intihar, ırkçı, ayrımcılık, nefret, küfür, argo, uyuşturucu, alkol, sigara, silah, bıçak, kan, yaralama, dövmek, işkence, kölelik, terörist, bomba, patlama, kaçırma, fidye, gasp, hırsızlık, dolandırıcılık, taciz, hap, zorbalık, istismar
 Bu kelimeler veya benzeri herhangi bir zararlı, saldırgan veya uygunsuz içerik, çocuklara yönelik bir uygulamada kesinlikle kabul edilemez. Konu değiştirirken "Bu konuda konuşmak istemiyorum. Matematik hakkında konuşalım mı?" gibi bir ifade kullanabilirsin.
 """
)

chat_session = model.start_chat()

# RAG Chatbot yükleme fonksiyonu
def load_rag_chatbot():
    global rag_chatbot
    
    if not RAG_AVAILABLE:
        print("⚠️ RAG Chatbot modülü mevcut değil, atlanıyor")
        return False
    
    try:
        print("RAG Chatbot yükleniyor...")
        rag_chatbot = AdvancedMatematikRAGChat()
        print("✅ RAG Chatbot başarıyla yüklendi!")
        return True
        
    except Exception as e:
        print(f"❌ RAG Chatbot yükleme hatası: {e}")
        return False

# Model yükleme fonksiyonu
def load_fine_tuned_model():
    global llm_model, llm_tokenizer
    
    try:
        print("Gemma RAG modeli yükleniyor...")
        
        # GPU memory optimizasyonu
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()
            gc.collect()
        
        # Gemma model'in tokenizer'ını yükle
        llm_tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_PATH)
        
        # Gemma base model yükle (CPU'ya yükle - güvenli)
        llm_model = AutoModelForCausalLM.from_pretrained(
            GEMMA_MODEL_PATH,
            device_map="cpu",  # CPU'ya yükle
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Padding token ayarla
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
        
        print("✅ Gemma RAG modeli başarıyla yüklendi!")
        return True
        
    except Exception as e:
        print(f"❌ Gemma model yükleme hatası: {e}")
        return False

# Fine-tuned model ile yanıt üretme fonksiyonu
def generate_llm_response(user_input):
    try:
        if llm_model is None or llm_tokenizer is None:
            return "Model henüz yüklenmedi. Lütfen bekleyin..."
        
        # Gemma prompt formatı
        prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
        
        # Tokenize ve CPU'da tut
        inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate (CPU model için optimize edilmiş)
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=llm_tokenizer.eos_token_id
            )
        
        # Decode response
        response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response - kullanıcı sorusunu tekrar etmesini önle
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1].strip()
            
            # Kullanıcı sorusunu response'dan tamamen çıkar
            user_input_lower = user_input.lower().strip()
            response_lower = response.lower()
            
            # Kullanıcı sorusunu response'dan çıkar
            if user_input_lower in response_lower:
                response = response.replace(user_input, "").strip()
                response = response.replace(user_input_lower, "").strip()
            
            # Response'u temizle - Gemma model için optimize edilmiş
            import re
            
            # Kullanıcı sorusunu response'dan çıkar
            if user_input.lower() in response.lower():
                response = response.replace(user_input, "").strip()
            
            # Gereksiz karakterleri temizle
            response = re.sub(r'\s+', ' ', response).strip()
            
            # Eğer response boşsa veya çok kısaysa, basit bir cevap ver
            if not response or len(response.strip()) < 3:
                response = "Bu soruyu cevaplayamıyorum, lütfen başka bir soru sorun."
        
        return response
        
    except Exception as e:
        print(f"LLM yanıt hatası: {e}")
        return "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/oyun')
def oyun():
    return render_template('oyun.html')

@app.route('/hikaye')
def hikaye():
    return render_template('hikaye.html')

@app.route('/api', methods=['POST'])
def api():
    try:
        user_input = request.json.get('user_input', '')
        if not user_input.strip():
            return jsonify({'bot_response': 'Lütfen bir mesaj yazın.'}), 400
        
        print(f"📝 Kullanıcı mesajı: {user_input}")
        
        # Önce RAG Chatbot'u dene
        if rag_chatbot is not None and RAG_AVAILABLE:
            try:
                print("🤖 RAG Chatbot kullanılıyor...")
                response = rag_chatbot.chat(user_input)
                print(f"✅ RAG yanıtı: {response[:100]}...")
                return jsonify({'bot_response': response})
            except Exception as e:
                print(f"❌ RAG Chatbot hatası: {e}")
                # RAG hatası durumunda diğer modellere geç
        
        # Lazy loading - Gemma RAG modeli ihtiyaç olduğunda yükle
        global llm_model, llm_tokenizer
        
        if llm_model is None:
            print("🔄 Gemma RAG modeli yükleniyor...")
            if load_fine_tuned_model():
                print("✅ Gemma RAG modeli başarıyla yüklendi!")
            else:
                print("❌ Gemma RAG modeli yüklenemedi, Gemini kullanılacak")
        
        # Gemma RAG model kullan
        if llm_model is not None and llm_tokenizer is not None:
            try:
                print("🤖 Gemma RAG model kullanılıyor...")
                response = generate_llm_response(user_input)
                
                # Matematik soruları için özel kontrol
                math_keywords = ["kaç", "topla", "çıkar", "çarp", "böl", "kalan", "yüzde", "kesir", "ondalık", "saat", "dakika", "metre", "kilogram"]
                is_math_question = any(keyword in user_input.lower() for keyword in math_keywords)
                
                # Eğer response çok karışık veya yanlışsa, Gemini kullan
                if (len(response) > 200 or 
                    user_input.lower() in response.lower() or
                    len(response.split()) < 2 or
                    (is_math_question and not any(char.isdigit() for char in response))):
                    
                    print("🔄 Gemma RAG model yanıtı uygun değil, Gemini kullanılıyor...")
                    raise Exception("Gemma RAG model yanıtı uygun değil")
                
                print(f"✅ Gemma RAG yanıtı: {response[:100]}...")
                return jsonify({'bot_response': response})
            except Exception as e:
                print(f"❌ Gemma RAG model hatası: {e}")
        
        # Fallback olarak Gemini kullan
        try:
            print("🤖 Gemini kullanılıyor...")
            response = chat_session.send_message(user_input)
            print(f"✅ Gemini yanıtı: {response.text[:100]}...")
            return jsonify({'bot_response': response.text})
        except Exception as e:
            print(f"❌ Gemini hatası: {e}")
            # Gemini quota hatası kontrolü
            if "429" in str(e) or "quota" in str(e).lower():
                print("⚠️ Gemini API kotası aşıldı, yerel model kullanılıyor...")
                # Basit bir yanıt döndür
                return jsonify({'bot_response': 'Şu anda çok yoğun bir dönemdeyiz. Lütfen biraz sonra tekrar deneyin veya farklı bir soru sorun.'})
            else:
                return jsonify({'bot_response': 'Üzgünüm, şu anda yanıt veremiyorum. Lütfen daha sonra tekrar deneyin.'})
            
    except Exception as e:
        print(f"❌ API genel hatası: {e}")
        return jsonify({'bot_response': 'Bir hata oluştu. Lütfen tekrar deneyin.'}), 500

# Konu anlatım botu için Gemma modeli
konu_anlatim_model = None
konu_anlatim_tokenizer = None

def load_konu_anlatim_model():
    global konu_anlatim_model, konu_anlatim_tokenizer
    
    try:
        print("Matematik QLoRA modeli yükleniyor...")
        
        # Import'ları başta yap
        import torch
        import gc
        
        # Memory'yi temizle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()
            gc.collect()
        
        # GPU memory optimizasyonu - HIZLI YÜKLEME
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Önce base model'in tokenizer'ını yükle
        konu_anlatim_tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_PATH)
        
        # Base model yükle - GÜVENLİ YÜKLEME AYARLARI
        konu_anlatim_model = AutoModelForCausalLM.from_pretrained(
            GEMMA_MODEL_PATH,
            device_map="auto",  # Otomatik device mapping
            torch_dtype=torch.float16,  # 16-bit precision - hız için
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_safetensors=True,  # Daha hızlı yükleme
            max_memory={0: "12GB"}  # GPU memory limiti - GÜVENLİ
        )
        
        # QLoRA adapter yükle - HIZLI
        konu_anlatim_model = PeftModel.from_pretrained(
            konu_anlatim_model, 
            MATEMATIK_QLORA_MODEL_PATH,
            torch_dtype=torch.float16  # 16-bit precision
        )
        
        # Padding token ayarla
        if konu_anlatim_tokenizer.pad_token is None:
            konu_anlatim_tokenizer.pad_token = konu_anlatim_tokenizer.eos_token
        
        print("✅ Matematik QLoRA modeli başarıyla yüklendi!")
        return True
        
    except Exception as e:
        print(f"❌ Matematik QLoRA modeli yükleme hatası: {e}")
        # Fallback: Base model kullan
        try:
            print("🔄 Base model kullanılıyor...")
            konu_anlatim_model = AutoModelForCausalLM.from_pretrained(
                GEMMA_MODEL_PATH,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            print("✅ Base model başarıyla yüklendi!")
            return True
        except Exception as e2:
            print(f"❌ Base model yükleme de başarısız: {e2}")
            return False

def generate_konu_anlatim_response(user_input):
    try:
        if konu_anlatim_model is None or konu_anlatim_tokenizer is None:
            return "Model henüz yüklenmedi. Lütfen bekleyin..."
        
        # Matematik odaklı prompt formatı - QLoRA model için optimize edilmiş
        prompt = f"<start_of_turn>user\nMatematik konusu hakkında soru: {user_input}<end_of_turn>\n<start_of_turn>model\n"
        
        # Tokenize ve device kontrolü
        inputs = konu_anlatim_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)  # Daha kısa
        
        # Device kontrolü - Model ve input aynı device'da olmalı
        model_device = next(konu_anlatim_model.parameters()).device
        for key in inputs:
            inputs[key] = inputs[key].to(model_device)
        
        # Generate (HIZLI - daha kısa token)
        with torch.no_grad():
            outputs = konu_anlatim_model.generate(
                **inputs,
                max_new_tokens=128,  # Çok daha kısa - hız için
                temperature=0.7,     # Biraz daha yaratıcı
                top_p=0.9,
                do_sample=True,
                pad_token_id=konu_anlatim_tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Tekrarı önle
                num_beams=1  # Greedy search - hız için
            )
        
        # Decode response
        response = konu_anlatim_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1].strip()
            
            # Kullanıcı sorusunu response'dan çıkar
            user_input_lower = user_input.lower().strip()
            response_lower = response.lower()
            
            if user_input_lower in response_lower:
                response = response.replace(user_input, "").strip()
                response = response.replace(user_input_lower, "").strip()
            
            # Response'u temizle
            import re
            response = re.sub(r'\s+', ' ', response).strip()
            
            # Matematik odaklı kontrol
            if not response or len(response.strip()) < 5:
                response = "Bu matematik konusunu anlatamıyorum, lütfen başka bir matematik konusu sorun."
        
        return response
        
    except Exception as e:
        print(f"Matematik konu anlatım yanıt hatası: {e}")
        return "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin."

@app.route('/api/konu-anlatim', methods=['POST'])
def konu_anlatim_api():
    try:
        user_input = request.json.get('user_input', '')
        if not user_input.strip():
            return jsonify({'bot_response': 'Lütfen bir matematik konusu sorun.'}), 400
        
        print(f"📚 Konu anlatım sorusu: {user_input}")
        
        # Lazy loading - QLoRA modeli ihtiyaç olduğunda yükle
        global konu_anlatim_model, konu_anlatim_tokenizer
        
        if konu_anlatim_model is None:
            print("🔄 QLoRA modeli yükleniyor...")
            if load_konu_anlatim_model():
                print("✅ QLoRA modeli başarıyla yüklendi!")
            else:
                print("❌ QLoRA modeli yüklenemedi, Gemini kullanılacak")
        
        # Matematik odaklı kontrol
        matematik_keywords = ["matematik", "toplama", "çıkarma", "çarpma", "bölme", "kesir", "ondalık", "yüzde", "geometri", "açı", "alan", "çevre", "hacim", "sayı", "problem", "işlem", "formül", "denklem", "eşitlik", "küme", "olasılık", "istatistik", "grafik", "tablo", "ölçü", "metre", "kilogram", "litre", "saat", "dakika", "saniye"]
        
        is_matematik_sorusu = any(keyword in user_input.lower() for keyword in matematik_keywords)
        
        # Önce QLoRA matematik modeli kullan
        if konu_anlatim_model is not None and konu_anlatim_tokenizer is not None:
            try:
                print("🤖 QLoRA matematik modeli kullanılıyor...")
                response = generate_konu_anlatim_response(user_input)
                
                # Matematik sorusu kontrolü
                if is_matematik_sorusu and (len(response) > 800 or 
                    user_input.lower() in response.lower() or
                    len(response.split()) < 5):
                    
                    print("🔄 QLoRA model yanıtı uygun değil, Gemini kullanılıyor...")
                    raise Exception("QLoRA model yanıtı uygun değil")
                
                print(f"✅ QLoRA yanıtı: {response[:100]}...")
                return jsonify({'bot_response': response})
            except Exception as e:
                print(f"❌ QLoRA model hatası: {e}")
        
        # Fallback olarak Gemini kullan
        try:
            print("🤖 Gemini kullanılıyor...")
            response = chat_session.send_message(user_input)
            print(f"✅ Gemini yanıtı: {response.text[:100]}...")
            return jsonify({'bot_response': response.text})
        except Exception as e:
            print(f"❌ Gemini hatası: {e}")
            # Gemini quota hatası kontrolü
            if "429" in str(e) or "quota" in str(e).lower():
                print("⚠️ Gemini API kotası aşıldı, yerel model kullanılıyor...")
                # Basit bir matematik yanıtı döndür
                return jsonify({'bot_response': 'Şu anda çok yoğun bir dönemdeyiz. Matematik konusunu daha sonra anlatabilirim. Lütfen biraz sonra tekrar deneyin.'})
            else:
                return jsonify({'bot_response': 'Üzgünüm, şu anda matematik konusunu anlatamıyorum. Lütfen daha sonra tekrar deneyin.'})
            
    except Exception as e:
        print(f"❌ Konu anlatım API genel hatası: {e}")
        return jsonify({'bot_response': 'Bir hata oluştu. Lütfen tekrar deneyin.'}), 500

@app.route('/api/generate_story', methods=['POST'])
def generate_story():
    prompt = request.json.get('prompt', "")
    suggestions = generate_story_suggestions(prompt=prompt, retries=3)  # `retries` burada int olarak belirleniyor
    if suggestions:
        return jsonify({"suggestions": suggestions})
    return jsonify({"error": "Hikaye başlangıcı oluşturulamadı."}), 500

# Hikaye devam önerileri için API endpoint
@app.route('/api/generate_continuation', methods=['POST'])
def generate_continuation():
    current_story = request.json.get('current_story', "")
    continuations = generate_continuation_options(current_story)
    if continuations:
        return jsonify({"continuations": continuations})
    return jsonify({"error": "Devam önerisi oluşturulamadı."}), 500

@app.route('/api/generate_final', methods=['POST'])
def generate_final():
    current_story = request.json.get('current_story', "")
    final_options = generate_final_step(current_story)
    if final_options:
        return jsonify({"final_options": final_options})  # JSON yanıtını 'final_options' olarak döndürün
    return jsonify({"error": "Hikaye tamamlama adımı oluşturulamadı."}), 500

# Türkçe metni sese çevirme (TTS) endpoint'i
@app.route('/api/tts', methods=['POST'])
def tts_endpoint():
    text = request.json.get('text', '')
    if not text.strip():
        return jsonify({'error': 'Metin boş'}), 400

    try:
        # gTTS Türkçe dili kullanılarak ses dosyası oluşturulur
        tts = gTTS(text=text, lang='tr')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)

        # Ses dosyasını Base64'e çevir
        audio_base64 = base64.b64encode(fp.read()).decode('utf-8')
        return jsonify({'audio_base64': audio_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # GPU memory tamamen temizle
        import torch
        import gc
        
        print("🧹 GPU Memory temizleniyor...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            print(f"📱 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # GPU memory ayarları - HIZLI KULLANIM
        if torch.cuda.is_available():
            # Memory'yi tamamen temizle
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()
            gc.collect()
            
            torch.cuda.set_per_process_memory_fraction(0.85)  # GPU memory'nin %85'ini kullan - GÜVENLİ
            print("⚡ GPU Memory kullanımı: %85 - GÜVENLİ MOD")
            
            # CUDA optimizasyonları
            torch.backends.cudnn.benchmark = True  # Hızlandırma
            torch.backends.cudnn.deterministic = False  # Hız için
            print("🚀 CUDA optimizasyonları aktif")
        
        # Model yükleme sırası - Sadece RAG Chatbot yükle
        models_loaded = 0
        
        # Sadece RAG Chatbot'u yükle (en önemli olan)
        print("\n🤖 RAG Chatbot yükleniyor...")
        if load_rag_chatbot():
            print("✅ RAG Chatbot başarıyla yüklendi!")
            models_loaded += 1
        else:
            print("❌ RAG Chatbot yüklenemedi, Gemini kullanılacak")
        
        # GPU memory'yi temizle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Diğer modeller lazy loading ile yüklenecek (ihtiyaç olduğunda)
        print("📝 Diğer modeller ihtiyaç olduğunda yüklenecek...")
        
        print(f"\n🎉 Toplam {models_loaded} model başarıyla yüklendi!")
        print("🚀 Uygulama başlatılıyor...")
        print("🌐 Web sayfası: http://localhost:5000")
        
        # Flask uygulamasını başlat
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
        
    except Exception as e:
        print(f"❌ Uygulama başlatma hatası: {e}")
        print("🔄 Basit modda başlatılıyor...")
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
        