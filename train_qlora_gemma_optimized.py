import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedQLoRATrainer:
    def __init__(self):
        self.base_model_path = "./gemma-2-9b-it-tr-new"
        self.output_dir = "./gemma-matematik-qlora-optimized"
        self.model = None
        self.tokenizer = None
        
    def load_model_and_tokenizer(self):
        """Model ve tokenizer yükle - optimize edilmiş"""
        logger.info("🚀 OPTİMİZE MODEL YÜKLEME...")
        
        # Memory temizle
        torch.cuda.empty_cache()
        
        try:
            # Tokenizer yükle
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # OPTİMİZE Quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # OPTİMİZE Model yükle - eager attention ile
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=bnb_config,
                device_map="cuda:0",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation="eager"  # ÖNERİLEN AYAR
            )
            
            # Model'i kbit training için hazırla
            self.model = prepare_model_for_kbit_training(self.model)
            
            logger.info("✅ Optimize model başarıyla yüklendi!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model yükleme hatası: {e}")
            return False
    
    def setup_lora_config(self):
        """OPTİMİZE LoRA config"""
        lora_config = LoraConfig(
            r=16,  # Biraz artırıldı
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return lora_config
    
    def load_and_prepare_data(self):
        """Veri yükle ve hazırla - TEST MODU"""
        logger.info("📊 Veri dosyaları yükleniyor... (TEST: Sadece ilk 100 örnek)")
        
        data_files = [
            "mat1konu.json", "mat2konu.json", "mat3konu.json", 
            "mat4konu.json", "mat5konu.json", "mat6konu.json",
            "mat7konu.json", "mat8_lgskonu.json", "mat8konu.json"
        ]
        
        all_data = []
        
        for file_name in data_files:
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # TEST: Sadece ilk 100 örnek
                limited_data = data[:100]
                all_data.extend(limited_data)
                
                logger.info(f"✅ {file_name} yüklendi ({len(limited_data)} örnek)")
                
            except Exception as e:
                logger.error(f"❌ {file_name} yükleme hatası: {e}")
        
        logger.info(f"✅ Toplam {len(all_data)} örnek hazırlandı (TEST MODU)")
        return all_data
    
    def tokenize_function(self, examples):
        """OPTİMİZE Tokenize fonksiyonu"""
        # Gemma formatında prompt hazırla
        prompts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"
            prompts.append(prompt)
        
        # Tokenize
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=512,  # Biraz artırıldı
            return_tensors="pt"
        )
        
        # Labels input_ids ile aynı
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def prepare_training_data(self, data):
        """Eğitim verisi hazırla"""
        logger.info("Eğitim verisi hazırlanıyor...")
        
        # Dataset oluştur
        dataset = Dataset.from_list(data)
        
        # Tokenize
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def setup_training_args(self):
        """OPTİMİZE Training arguments"""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=2,  # Biraz artırıldı
            per_device_train_batch_size=2,  # Biraz artırıldı
            gradient_accumulation_steps=4,  # Optimize
            warmup_steps=50,  # Biraz artırıldı
            learning_rate=2e-4,  # Optimize
            logging_steps=10,  # Daha sık
            save_steps=200,  # Daha sık
            save_strategy="steps",
            dataloader_num_workers=0,  # Windows için
            max_grad_norm=1.0,
            logging_first_step=True,
            remove_unused_columns=False,  # Önemli
            gradient_checkpointing=True,  # Memory için
            fp16=True,  # Mixed precision
            optim="paged_adamw_8bit",  # Memory optimize
            lr_scheduler_type="cosine",  # Daha iyi convergence
            report_to=None,  # Wandb kapatıldı
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=False,  # Windows için
        )
        
        return training_args
    
    def train(self):
        """Ana eğitim fonksiyonu"""
        logger.info("🚀 OPTİMİZE QLoRA EĞİTİMİ BAŞLATILIYOR...")
        
        # Model yükle
        if not self.load_model_and_tokenizer():
            return False
        
        # LoRA config
        self.setup_lora_config()
        
        # Veri yükle
        data = self.load_and_prepare_data()
        if not data:
            logger.error("❌ Veri yüklenemedi!")
            return False
        
        # Eğitim verisi hazırla
        train_dataset = self.prepare_training_data(data)
        
        # Training arguments
        training_args = self.setup_training_args()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Eğitim başlat
        logger.info("🚀 OPTİMİZE EĞİTİM BAŞLIYOR...")
        trainer.train()
        
        # Model kaydet
        logger.info("💾 Model kaydediliyor...")
        trainer.save_model()
        
        logger.info(f"✅ OPTİMİZE EĞİTİM TAMAMLANDI! Model {self.output_dir} dizinine kaydedildi")
        return True

def main():
    print("🐇 OPTİMİZE QLoRA EĞİTİMİ")
    print("=" * 60)
    print("⚡ OPTİMİZE MODU: Terminal önerileri kullanılarak")
    print("📊 2 epoch, optimize batch, eager attention")
    print("=" * 60)
    
    trainer = OptimizedQLoRATrainer()
    
    try:
        success = trainer.train()
        if success:
            print("\n🎉 OPTİMİZE EĞİTİM BAŞARILI!")
            print(f"📁 Model: {trainer.output_dir}")
        else:
            print("\n❌ Eğitim başarısız!")
    except Exception as e:
        print(f"\n❌ Ana hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 