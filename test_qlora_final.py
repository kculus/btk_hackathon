import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import gc

class QLoRATester:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Model yükle"""
        print("🚀 Model yükleniyor...")
        
        # Memory temizle
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("./gemma-2-9b-it-tr-new", trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # Base model
            base_model = AutoModelForCausalLM.from_pretrained(
                "./gemma-2-9b-it-tr-new",
                quantization_config=bnb_config,
                device_map="cuda:0",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            
            # LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, "./gemma-matematik-qlora")
            self.model.eval()
            
            print("✅ Model başarıyla yüklendi!")
            return True
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            return False
    
    def generate_response(self, question):
        """Cevap üret"""
        try:
            # Prompt hazırla
            prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Sadece model cevabını çıkar
            if "<start_of_turn>model" in full_response:
                response = full_response.split("<start_of_turn>model")[-1].strip()
                # End token'ı kaldır
                if "<end_of_turn>" in response:
                    response = response.split("<end_of_turn>")[0].strip()
                return response
            else:
                return full_response
            
        except Exception as e:
            return f"Hata: {e}"
    
    def test_examples(self):
        """Test örnekleri"""
        test_questions = [
            "Doğal sayılar nedir?",
            "Kesirler nasıl gösterilir?",
            "Alan hesaplama nasıl yapılır?",
            "Ondalık sayılar nedir?",
            "Geometrik şekiller nelerdir?"
        ]
        
        print("\n🧪 TEST ÖRNEKLERİ:")
        print("=" * 50)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Soru: {question}")
            response = self.generate_response(question)
            print(f"Cevap: {response}")
            print("-" * 40)
    
    def interactive_test(self):
        """İnteraktif test"""
        print("\n💬 İNTERAKTİF TEST MODU")
        print("Çıkmak için 'quit' yazın")
        print("=" * 50)
        
        while True:
            try:
                question = input("\nSoru: ")
                if question.lower() in ['quit', 'exit', 'çık']:
                    break
                
                print("🤔 Düşünüyor...")
                response = self.generate_response(question)
                print(f"💡 Cevap: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Hata: {e}")
    
    def cleanup(self):
        """Memory temizle"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

def main():
    print("🐇 QLoRA Matematik Model Test")
    print("=" * 50)
    
    tester = QLoRATester()
    
    try:
        if tester.load_model():
            # Test örnekleri
            tester.test_examples()
            
            # İnteraktif test
            tester.interactive_test()
        else:
            print("❌ Model yüklenemedi!")
    except Exception as e:
        print(f"❌ Ana hata: {e}")
    finally:
        tester.cleanup()
    
    print("\n✅ Test tamamlandı!")

if __name__ == "__main__":
    main() 