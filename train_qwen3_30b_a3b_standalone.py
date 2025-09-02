#!/usr/bin/env python3
"""
Standalone training script for Qwen3-30B-A3B model using Unsloth
This script demonstrates fine-tuning Qwen3-30B-A3B with LoRA adapters
"""

import os
import time
import argparse
import torch
import pandas as pd
from datasets import load_dataset, Dataset, Value
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from trl import SFTTrainer, SFTConfig
import triton.profiler as proton


def prepare_dataset(tokenizer, num_samples=1000):
    print("Loading FineTome-100k dataset...")
    dataset = load_dataset("mlabonne/FineTome-100k", split="train")

    # 1) Standardize to role/content structure
    dataset = standardize_data_formats(dataset)

    # 2) Use Qwen3 expected template (gemma-3)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    # 3) Compatible column names: some have "messages", some have "conversations" after standardization
    col = "conversations" if "conversations" in dataset.column_names else (
          "messages" if "messages" in dataset.column_names else None)
    if col is None:
        raise KeyError(f"Expected 'conversations' or 'messages' in dataset columns, got {dataset.column_names}")

    # 4) Generate string format training text (explicit tokenize=False)
    def apply_chat_template_func(examples):
        convs = examples[col]            # list[list[{"role","content"}]]
        texts = [
            tokenizer.apply_chat_template(
                c,
                tokenize=False,              # Key: ensure returning string not input_ids
                add_generation_prompt=False  # Training SFT generally doesn't add inference prompt
            )
            for c in convs
        ]
        return {"text": texts}

    dataset = dataset.map(
        apply_chat_template_func,
        batched=True,
        remove_columns=dataset.column_names
    )

    # 5) Filter non-string/blank samples to avoid tokenizer receiving None or ""
    dataset = dataset.filter(lambda x: isinstance(x["text"], str) and len(x["text"].strip()) > 0)

    # 6) Explicitly declare column type as string (optional but more stable)
    dataset = dataset.cast_column("text", Value("string"))

    # 7) Truncate samples
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Dataset prepared with {len(dataset)} samples")
    return dataset, tokenizer


def initialize_model():
    """
    Initialize Qwen3-30B-A3B model with LoRA adapters
    
    Returns:
        model: Model with LoRA adapters
        tokenizer: Tokenizer for the model
    """
    print("Initializing Qwen3-30B-A3B model...")
    
    # Load base model and tokenizer
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/Qwen3-30B-A3B",
        max_seq_length=2048,
        load_in_4bit=False,  # Set to True for 4-bit quantization
        load_in_8bit=False,  # Set to True for 8-bit quantization
        full_finetuning=False,  # We'll use LoRA instead
        # cache_dir="/scratch/jlee436/unsloth/model"
    )
    
    print("Adding LoRA adapters...")
    # Configure LoRA adapters for parameter-efficient fine-tuning
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Qwen is text-only
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,  # LoRA rank
        lora_alpha=8,  # LoRA scaling parameter
        lora_dropout=0,  # LoRA dropout
        bias="none",  # Bias training strategy
        random_state=3407,  # For reproducibility
    )
    
    print(f"Model initialized with {model.num_parameters():,} parameters")
    return model, tokenizer


def create_trainer(model, tokenizer, train_dataset):
    """
    Create SFTTrainer with training configuration
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: The training dataset
    
    Returns:
        SFTTrainer instance
    """
    print("Creating trainer with SFT configuration...")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,  # Add validation dataset here if needed
        args=SFTConfig(
            # Dataset configuration
            dataset_text_field="text",
            
            # Batch size and accumulation
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
            
            # Training steps
            warmup_steps=5,
            max_steps=30,  # Increase for longer training
            
            # Learning rate and optimizer
            learning_rate=2e-4,
            optim="adamw_8bit",  # 8-bit AdamW for memory efficiency
            weight_decay=0.01,
            lr_scheduler_type="linear",
            
            # Logging
            logging_steps=15,
            report_to="none",  # Change to "wandb" or "tensorboard" for logging
            
            # Other settings
            seed=3407,
        ),
    )
    
    # Apply response-only training for Qwen3 format (using gemma-3 template parts)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )
    
    return trainer


def train_with_profiling(trainer, profiling_mode="none", model_name="qwen3_30b_a3b"):
    """
    Train the model with optional profiling
    
    Args:
        trainer: The SFTTrainer instance
        profiling_mode: One of "none", "proton", or "torch"
        model_name: Name for profiling output files
    """
    print(f"\n{'='*50}")
    print(f"Starting training with profiling mode: {profiling_mode}")
    print(f"{'='*50}\n")
    
    if profiling_mode == "proton":
        # Use Triton Proton profiler
        session_id = proton.start(name=f"qwen3_30b_a3b_{model_name}", context="shadow")

        # Measure only the training loop time
        start_time = time.time()
        print(f"START_TRAINING_LOOP: {start_time}")
        trainer.train()
        end_time = time.time()
        print(f"END_TRAINING_LOOP: {end_time}")
        print(f"\nTraining loop completed in {end_time - start_time:.2f} seconds")
        proton.finalize(session_id)
        print(f"Proton profile saved as qwen3_30b_a3b_{model_name}.hatchet")
        
    elif profiling_mode == "torch":
        # Use PyTorch profiler
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            # Measure only the training loop time
            start_time = time.time()
            print(f"START_TRAINING_LOOP: {start_time}")
            trainer.train()
            end_time = time.time()
            print(f"END_TRAINING_LOOP: {end_time}")
            print(f"\nTraining loop completed in {end_time - start_time:.2f} seconds")
        
        trace_file = f"qwen3_30b_a3b_trace_{model_name}.json"
        prof.export_chrome_trace(trace_file)
        print(f"PyTorch profile saved as {trace_file}")
        
    else:
        # Standard training without profiling
        # Measure only the training loop time
        start_time = time.time()
        print(f"START_TRAINING_LOOP: {start_time}")
        trainer.train()
        end_time = time.time()
        print(f"END_TRAINING_LOOP: {end_time}")
        print(f"\nTraining loop completed in {end_time - start_time:.2f} seconds")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-30B-A3B model using Unsloth and LoRA"
    )
    parser.add_argument(
        "--profiling",
        type=str,
        default="none",
        choices=["none", "proton", "torch"],
        help="Profiling mode: none (default), proton, or torch"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of training samples to use (default: 1000)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum number of training steps (default: 30)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device training batch size (default: 2)"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Enable 4-bit quantization for reduced memory usage"
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Enable 8-bit quantization for reduced memory usage"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Qwen3-30B-A3B Fine-tuning with Unsloth")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Profiling mode: {args.profiling}")
    print(f"  - Number of samples: {args.num_samples}")
    print(f"  - Max training steps: {args.max_steps}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - 4-bit quantization: {args.load_in_4bit}")
    print(f"  - 8-bit quantization: {args.load_in_8bit}")
    print("="*60 + "\n")
    
    # Initialize model and tokenizer
    model, tokenizer = initialize_model()
    
    # Prepare dataset
    train_dataset, tokenizer = prepare_dataset(tokenizer, args.num_samples)
    
    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset)
    
    # Override max_steps if specified
    if args.max_steps != 30:
        trainer.args.max_steps = args.max_steps
    if args.batch_size != 2:
        trainer.args.per_device_train_batch_size = args.batch_size
    
    # Start training with profiling
    train_with_profiling(trainer, args.profiling, "standalone")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    # Optional: Save the model
    # print("Saving model...")
    # model.save_pretrained("./qwen3_30b_a3b_finetuned")
    # tokenizer.save_pretrained("./qwen3_30b_a3b_finetuned")


if __name__ == "__main__":
    main()