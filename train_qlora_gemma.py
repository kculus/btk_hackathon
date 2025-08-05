import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import logging
from tqdm import tqdm
import gc

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatematikQLoRATrainer:
    def __init__(self):
        self.model_name = "./gemma-2-9b-it-tr-new"
        self.output_dir = "./gemma-matematik-qlora"
        self.data_files = [
            "mat1konu.json",
            "mat2konu.json", 
            "mat3konu.json",
            "mat4konu.json",
            "mat5konu.json",
            "mat6konu.json",
            "mat7konu.json",
            "mat8_lgskonu.json",
            "mat8konu.json"
        ]
        
        # QLoRA konfigürasyonu - TEST İÇİN KÜÇÜK
        self.lora_config = LoraConfig(
            r=8,  # Rank - test için küçük
            lora_alpha=16,  # Alpha scaling - test için küçük
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Quantization konfigürasyonu
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        
    def load_model_and_tokenizer(self):
        """Model ve tokenizer'ı yükle"""
        logger.info("Model ve tokenizer yükleniyor...")
        
        try:
            # Tokenizer yükle
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Padding token ayarla
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Model yükle (4-bit quantization ile)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self.bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            # Model'i kbit training için hazırla
            self.model = prepare_model_for_kbit_training(self.model)
            
            # QLoRA adapter'ını ekle
            self.model = get_peft_model(self.model, self.lora_config)
            
            logger.info("✅ Model ve tokenizer başarıyla yüklendi!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model yükleme hatası: {e}")
            return False
    
    def load_and_prepare_data(self):
        """Veri dosyalarını yükle ve hazırla - TEST İÇİN SADECE İLK 50 ÖRNEK"""
        logger.info("Veri dosyaları yükleniyor... (TEST: Sadece ilk 50 örnek)")
        
        all_data = []
        
        for file_path in self.data_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # TEST: Sadece ilk 50 örnek al
                        data = data[:50] if len(data) > 50 else data
                        all_data.extend(data)
                        logger.info(f"✅ {file_path} yüklendi ({len(data)} örnek)")
                except Exception as e:
                    logger.error(f"❌ {file_path} yükleme hatası: {e}")
            else:
                logger.warning(f"⚠️ {file_path} bulunamadı")
        
        if not all_data:
            logger.error("❌ Hiç veri yüklenemedi!")
            return False
        
        # Veriyi Gemma formatına dönüştür
        formatted_data = []
        for item in all_data:
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            
            # Gemma chat formatı
            formatted_text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"
            formatted_data.append({"text": formatted_text})
        
        # Dataset oluştur
        self.train_dataset = Dataset.from_list(formatted_data)
        
        logger.info(f"✅ Toplam {len(formatted_data)} örnek hazırlandı (TEST MODU)")
        return True
    
    def tokenize_function(self, examples):
        """Veriyi tokenize et"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=256,  # TEST: Daha kısa
            return_tensors="pt"
        )
    
    def prepare_training_data(self):
        """Eğitim verisini hazırla"""
        logger.info("Eğitim verisi hazırlanıyor...")
        
        # Tokenize et
        tokenized_dataset = self.train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.train_dataset.column_names
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        return tokenized_dataset, data_collator
    
    def train(self):
        """Eğitimi başlat - TEST İÇİN HIZLI"""
        logger.info("🚀 QLoRA TEST EĞİTİMİ BAŞLATILIYOR...")
        
        # Model ve tokenizer yükle
        if not self.load_model_and_tokenizer():
            return False
        
        # Veri yükle
        if not self.load_and_prepare_data():
            return False
        
        # Eğitim verisini hazırla
        tokenized_dataset, data_collator = self.prepare_training_data()
        
        # Training arguments - TEST İÇİN HIZLI
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=1,  # TEST: Sadece 1 epoch
            per_device_train_batch_size=1,  # TEST: Küçük batch
            gradient_accumulation_steps=2,  # TEST: Daha az
            warmup_steps=10,  # TEST: Daha az
            learning_rate=3e-4,  # TEST: Biraz daha yüksek
            fp16=True,
            logging_steps=5,  # TEST: Daha sık log
            save_steps=100,  # TEST: Daha sık kaydet
            save_strategy="steps",
            report_to=None,  # Tensorboard'u kapat
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            # TEST: Daha hızlı eğitim için
            dataloader_num_workers=0,
            max_grad_norm=1.0,
            logging_first_step=True
        )
        
        # Trainer oluştur
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Eğitimi başlat
        logger.info("🚀 TEST EĞİTİMİ BAŞLIYOR...")
        trainer.train()
        
        # Modeli kaydet
        logger.info("💾 Model kaydediliyor...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"✅ TEST EĞİTİMİ TAMAMLANDI! Model {self.output_dir} dizinine kaydedildi")
        return True
    
    def cleanup(self):
        """Memory temizle"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

def main():
    """Ana fonksiyon"""
    print("🐇 Matematik QLoRA TEST EĞİTİMİ Başlatılıyor...")
    print("=" * 60)
    print("⚡ TEST MODU: Hızlı eğitim için optimize edildi")
    print("📊 1 epoch, küçük batch, sınırlı veri")
    print("=" * 60)
    
    trainer = MatematikQLoRATrainer()
    
    try:
        success = trainer.train()
        if success:
            print("\n🎉 TEST EĞİTİMİ BAŞARILI!")
            print(f"📁 Model: {trainer.output_dir}")
            print("🧪 Şimdi test_qlora_model.py ile test edebilirsin!")
        else:
            print("\n❌ TEST EĞİTİMİ BAŞARISIZ!")
    except Exception as e:
        print(f"\n💥 Hata: {e}")
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main() 