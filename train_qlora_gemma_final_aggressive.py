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

class AggressiveQLoRATrainer:
    def __init__(self):
        self.base_model_path = "./gemma-2-9b-it-tr-new"
        self.output_dir = "./gemma-matematik-qlora-aggressive"
        self.model = None
        self.tokenizer = None
        
    def load_model_and_tokenizer(self):
        """Model ve tokenizer yükle - AGGRESSIVE MODU"""
        logger.info("🚀 AGGRESSIVE MODEL YÜKLEME...")
        
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
            
            # AGGRESSIVE Quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float32
            )
            
            # AGGRESSIVE Model yükle
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=bnb_config,
                device_map="cuda:0",
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                attn_implementation="eager"
            )
            
            # Model'i kbit training için hazırla
            self.model = prepare_model_for_kbit_training(self.model)
            
            logger.info("✅ Aggressive model başarıyla yüklendi!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model yükleme hatası: {e}")
            return False
    
    def setup_lora_config(self):
        """AGGRESSIVE LoRA config - DÜŞÜK LOSS İÇİN"""
        lora_config = LoraConfig(
            r=16,  # AGGRESSIVE - daha fazla parametre
            lora_alpha=32,  # AGGRESSIVE
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Tüm modüller
            lora_dropout=0.05,  # Minimal dropout
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return lora_config
    
    def load_and_prepare_data(self):
        """Veri yükle ve hazırla - AGGRESSIVE MODU"""
        logger.info("📊 Veri dosyaları yükleniyor... (AGGRESSIVE: Tüm örnekler)")
        
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
                
                # AGGRESSIVE: Tüm örnekleri kullan
                all_data.extend(data)
                
                logger.info(f"✅ {file_name} yüklendi ({len(data)} örnek)")
                
            except Exception as e:
                logger.error(f"❌ {file_name} yükleme hatası: {e}")
        
        logger.info(f"✅ Toplam {len(all_data)} örnek hazırlandı (AGGRESSIVE MODU)")
        return all_data
    
    def tokenize_function(self, examples):
        """AGGRESSIVE Tokenize fonksiyonu - DÜŞÜK LOSS İÇİN"""
        # Gemma formatında prompt hazırla
        prompts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"
            prompts.append(prompt)
        
        # Tokenize - AGGRESSIVE
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=512,  # AGGRESSIVE - daha uzun sequence
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
        """AGGRESSIVE Training arguments - DÜŞÜK LOSS İÇİN"""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=5,  # AGGRESSIVE - daha fazla epoch
            per_device_train_batch_size=1,  # Memory için düşürüldü
            gradient_accumulation_steps=4,  # AGGRESSIVE
            warmup_steps=50,  # AGGRESSIVE
            learning_rate=5e-6,  # AGGRESSIVE - çok düşük learning rate
            logging_steps=10,  # Daha sık
            save_steps=200,  # Daha sık
            save_strategy="steps",
            dataloader_num_workers=0,  # Windows için
            max_grad_norm=0.3,  # AGGRESSIVE - daha sıkı gradient clipping
            logging_first_step=True,
            remove_unused_columns=False,  # Önemli
            gradient_checkpointing=False,  # NaN için kapatıldı
            fp16=False,  # NaN için kapatıldı
            optim="adamw_torch",  # NaN için değiştirildi
            lr_scheduler_type="cosine",  # AGGRESSIVE - cosine scheduler
            report_to=None,  # Wandb kapatıldı
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=False,  # Windows için
            weight_decay=0.01,  # AGGRESSIVE - regularization
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
        )
        
        return training_args
    
    def train(self):
        """Ana eğitim fonksiyonu"""
        logger.info("🚀 AGGRESSIVE QLoRA EĞİTİMİ BAŞLATILIYOR (DÜŞÜK LOSS)...")
        
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
        logger.info("🚀 AGGRESSIVE EĞİTİM BAŞLIYOR (DÜŞÜK LOSS)...")
        trainer.train()
        
        # Model kaydet
        logger.info("💾 Aggressive model kaydediliyor...")
        trainer.save_model()
        
        logger.info(f"✅ AGGRESSIVE EĞİTİM TAMAMLANDI! Model {self.output_dir} dizinine kaydedildi")
        return True

def main():
    print("🐇 AGGRESSIVE QLoRA EĞİTİMİ (DÜŞÜK LOSS)")
    print("=" * 60)
    print("🎯 AGGRESSIVE MODU: Loss düşürme odaklı")
    print("📊 5 epoch, tüm veri, agresif ayarlar")
    print("=" * 60)
    
    trainer = AggressiveQLoRATrainer()
    
    try:
        success = trainer.train()
        if success:
            print("\n🎉 AGGRESSIVE EĞİTİM BAŞARILI!")
            print(f"📁 Aggressive Model: {trainer.output_dir}")
        else:
            print("\n❌ Eğitim başarısız!")
    except Exception as e:
        print(f"\n❌ Ana hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 