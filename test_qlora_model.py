import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QLoRATester:
    def __init__(self):
        self.base_model_path = "./gemma-2-9b-it-tr-new"
        self.adapter_path = "./gemma-matematik-qlora"
        self.model = None
        self.tokenizer = None
        
        # Memory temizle
        torch.cuda.empty_cache()
        gc.collect()
    
    def load_model(self):
        """Fine-tuned modeli yükle - GPU'ya optimize edilmiş"""
        logger.info("Fine-tuned model yükleniyor...")
        
        try:
            # Memory temizle
            torch.cuda.empty_cache()
            gc.collect()
            
            # Tokenizer yükle
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Quantization config - GPU memory için optimize
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # Base model yükle - GPU'ya optimize
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=bnb_config,
                device_map="cuda:0",  # Direkt GPU'ya
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            
            # LoRA adapter'ını yükle
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            
            # Model'i eval moduna al
            self.model.eval()
            
            logger.info("✅ Fine-tuned model GPU'da başarıyla yüklendi!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model yükleme hatası: {e}")
            return False
    
    def generate_response(self, instruction, max_length=128):
        """Cevap üret - GPU optimize"""
        try:
            # Memory temizle
            torch.cuda.empty_cache()
            
            # Gemma formatında prompt hazırla
            prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
            
            # Tokenize et
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            
            # GPU'ya taşı
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate - GPU optimize
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Assistant response'ını çıkar
            if "<start_of_turn>model" in response:
                response = response.split("<start_of_turn>model")[-1].strip()
                # End token'ı kaldır
                if "<end_of_turn>" in response:
                    response = response.split("<end_of_turn>")[0].strip()
            
            # Memory temizle
            del outputs
            torch.cuda.empty_cache()
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Cevap üretme hatası: {e}")
            return f"Hata: {e}"
    
    def quick_test(self):
        """Hızlı test örnekleri"""
        test_cases = [
            "Doğal sayılar nedir?",
            "Kesirler nasıl gösterilir?",
            "Alan hesaplama nasıl yapılır?"
        ]
        
        print("\n🧪 HIZLI TEST ÖRNEKLERİ:")
        print("=" * 50)
        
        for i, question in enumerate(test_cases, 1):
            print(f"\n{i}. Soru: {question}")
            response = self.generate_response(question)
            print(f"Cevap: {response}")
            print("-" * 30)
    
    def cleanup(self):
        """Memory temizle"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

def main():
    print("🧪 QLoRA HIZLI TEST Başlatılıyor...")
    print("=" * 50)
    
    tester = QLoRATester()
    
    try:
        if tester.load_model():
            tester.quick_test()
            
            # İnteraktif test
            print("\n💬 İnteraktif Test Modu:")
            print("Çıkmak için 'quit' yazın")
            print("-" * 30)
            
            while True:
                try:
                    user_input = input("\nSoru: ")
                    if user_input.lower() in ['quit', 'exit', 'çık']:
                        break
                    
                    response = tester.generate_response(user_input)
                    print(f"Cevap: {response}")
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Hata: {e}")
    except Exception as e:
        print(f"Ana hata: {e}")
    finally:
        tester.cleanup()
    
    print("\n✅ Test tamamlandı!")

if __name__ == "__main__":
    main() 