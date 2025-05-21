import os
import torch
import math
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
import mlflow
from mlflo import setup_mlflow, log_parameters, log_metrics

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)


torch.manual_seed(42)
# Configuration
MODEL_NAME = "openai-community/gpt2"
OUTPUT_DIR = "models/drug-llm"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 3e-4
NUM_EPOCHS = 3
MAX_LENGTH = 512


mlflow.start_run()
experiment_id = setup_mlflow()
log_parameters(LORA_R, LORA_ALPHA, LORA_DROPOUT, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, NUM_EPOCHS, MAX_LENGTH)

# Load datasets
dataset = load_dataset('json', data_files={
    'train': 'data/processed/train.jsonl',
    'validation': 'data/evaluation/eval.jsonl'
})

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
# GPT-2 doesn't have a pad token by default
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenize the datasets
def preprocess_function(examples):
    """Preprocess the dataset by tokenizing and formatting."""
    texts = [
        f"Instruction: {instruction}\nResponse: {response}"
        for instruction, response in zip(examples['input'], examples['output'])
    ]
    
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # Create labels (same as input_ids for causal LM)
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['input', 'output']
)

# Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ====== EVALUATE BASE MODEL FIRST ======
print('Loading the base model for evaluation...')
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    use_cache=False,
    device_map="auto"
)
base_model.resize_token_embeddings(len(tokenizer))

# Create evaluation args for the base model
base_eval_args = TrainingArguments(
    output_dir="./results/base_model_eval",
    per_device_eval_batch_size=BATCH_SIZE,
    remove_unused_columns=False,
    report_to="none",  # Disable reporting
)

# Create Trainer for evaluation only
base_trainer = Trainer(
    model=base_model,
    args=base_eval_args,
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

# Evaluate base model
print('Evaluating base model...')
base_metrics = base_trainer.evaluate()
base_eval_loss = base_metrics['eval_loss']
base_perplexity = math.exp(base_eval_loss)
print(f"\nBase Model Eval Loss: {base_eval_loss:.4f}")
print(f"Base Model Perplexity: {base_perplexity:.4f}")

# Free up memory
del base_model
del base_trainer
torch.cuda.empty_cache()  # If using GPU

# ====== NOW CONTINUE WITH FINE-TUNING ======
print('Loading model for fine-tuning...')
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    use_cache=False,
    device_map="auto"
)
model.resize_token_embeddings(len(tokenizer))

# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["c_attn"],
    fan_in_fan_out=False
)

model = get_peft_model(model, peft_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=100,
    save_steps=100,
    load_best_model_at_end=True,
    report_to="tensorboard",
    fp16=True,
    gradient_checkpointing=False,
    warmup_ratio=0.1,
    weight_decay=0.01,
    remove_unused_columns=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

# Train model
print('Fine-tuning model...')
trainer.train()

# Evaluate fine-tuned model
print('Evaluating fine-tuned model...')
ft_metrics = trainer.evaluate()
ft_eval_loss = ft_metrics['eval_loss']
ft_perplexity = math.exp(ft_eval_loss)

# Print comparison
print("\n===== EVALUATION RESULTS COMPARISON =====")
print(f"Base Model - Eval Loss: {base_eval_loss:.4f}, Perplexity: {base_perplexity:.4f}")
print(f"Fine-tuned - Eval Loss: {ft_eval_loss:.4f}, Perplexity: {ft_perplexity:.4f}")
print(f"Improvement  - Loss: {base_eval_loss - ft_eval_loss:.4f} ({(1 - ft_eval_loss/base_eval_loss) * 100:.2f}%)")
print(f"Improvement  - Perplexity: {base_perplexity - ft_perplexity:.4f} ({(1 - ft_perplexity/base_perplexity) * 100:.2f}%)")
log_metrics(ft_eval_loss, ft_perplexity)
# Save the fine-tuned model
model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print("Fine-tuning completed and model saved!")