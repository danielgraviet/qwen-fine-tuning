import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
OUTPUT_DIR = "./qwen-finetuned"

# 4-bit quantization (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# LoRA config — targets the attention projection layers
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# completion_only_loss=True computes loss only on the completion column,
# not the prompt. Works with any chat template without needing {% generation %} markers.
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,     # effective batch size = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    max_length=512,
    completion_only_loss=True,
)

def format_prompt_completion(example, tokenizer):
    """Format as prompt/completion so completion_only_loss knows where to mask."""
    prompt_messages = [{"role": "user", "content": example["instruction"]}]
    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    completion = example["output"] + tokenizer.eos_token
    return {"prompt": prompt, "completion": completion}


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )

    dataset = load_dataset("tatsu-lab/alpaca", split="train[:2000]")
    dataset = dataset.map(
        lambda ex: format_prompt_completion(ex, tokenizer),
        remove_columns=dataset.column_names,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
