import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import gc

def test_model():
    print("🧪 Basit QLoRA Test")
    print("=" * 30)
    
    # Memory temizle
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        # Tokenizer yükle
        print("1. Tokenizer yükleniyor...")
        tokenizer = AutoTokenizer.from_pretrained("./gemma-2-9b-it-tr-new", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer yüklendi")
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Base model yükle
        print("2. Base model yükleniyor...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "./gemma-2-9b-it-tr-new",
            quantization_config=bnb_config,
            device_map="cuda:0",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        print("✅ Base model yüklendi")
        
        # LoRA adapter yükle
        print("3. LoRA adapter yükleniyor...")
        model = PeftModel.from_pretrained(base_model, "./gemma-matematik-qlora")
        model.eval()
        print("✅ LoRA adapter yüklendi")
        
        # Basit test
        print("4. Test sorusu soruluyor...")
        prompt = "<start_of_turn>user\nDoğal sayılar nedir?<end_of_turn>\n<start_of_turn>model\n"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        print("5. Cevap üretiliyor...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,  # Çok kısa
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1].strip()
            if "<end_of_turn>" in response:
                response = response.split("<end_of_turn>")[0].strip()
        
        print(f"✅ Cevap: {response}")
        
        # Memory temizle
        del model, base_model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()
        
        print("🎉 Test başarılı!")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model() 