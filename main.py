from transformers import (AutoTokenizer, AutoModelForCausalLM, 
                          BitsAndBytesConfig)
import transformers
import torch
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def run_inference(
        model, tokenizer,
        question="Can you described the vision pro of apple in 3 sentences ?"):
    model.config.use_cache = True
    model.eval()
    print("Question:")
    print(question)
    encoding = tokenizer(question, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        output = model.generate(input_ids=encoding.input_ids,
                                attention_mask=encoding.attention_mask,
                                max_new_tokens=100, do_sample=True,
                                temperature=0.000001,
                                eos_token_id=tokenizer.eos_token_id, top_k=0)
        print("Answer:")
        print(tokenizer.decode(output[0], skip_special_tokens=True))


def load_pretrained_model(model_id="vilsonrodrigues/falcon-7b-instruct-sharded"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map={"": 0},
        trust_remote_code=True)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def prepare_dataset(tokenizer, dataset_file):

    data = load_dataset("json", data_files=dataset_file,
                        field="data")

    train_dataset = data['train'].select(range(60)).map(
        lambda x: {"input_text": x['question'] + "\n" +
                   x['best_answer']})

    train_encodings = tokenizer(
        train_dataset['input_text'],truncation=True, padding=True,
        max_length=256, return_tensors='pt')

    train_dataset = TextDataset(train_encodings)

    return train_dataset


def launch_training(model, tokenizer, train_dataset):
    training_configuration = transformers.TrainingArguments(
            num_train_epochs=30,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            warmup_ratio=0.05,
            # max_steps=100,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
            lr_scheduler_type='cosine',
        )
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_configuration,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False),
    )
    model.config.use_cache = False
    trainer.train()

    trainer.save_model("finetuned_falcon")


def main(dataset_file="visionpro_dataset.json"):
    model, tokenizer = load_pretrained_model()
    train_dataset = prepare_dataset(tokenizer, dataset_file)

    launch_training(model, tokenizer, train_dataset)

    run_inference(model, tokenizer)


if __name__ == "__main__":
    main()
