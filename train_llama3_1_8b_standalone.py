#!/usr/bin/env python3
"""
Standalone training script for Llama-3.1-8B model using Unsloth
This script demonstrates fine-tuning Llama-3.1-8B with LoRA adapters
"""

import os
import time
import argparse
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from trl import SFTTrainer, SFTConfig
import triton.profiler as proton


def prepare_dataset(tokenizer, num_samples=1000):
    print("Loading FineTome-100k dataset...")
    dataset = load_dataset("mlabonne/FineTome-100k", split="train")

    # 1) Standardize ShareGPT -> role/content
    dataset = standardize_sharegpt(dataset)  # No need to pass tokenizer

    # 2) Set Llama-3.1 chat template
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # 3) Apply template to each conversation (note: for individual conv, not entire column)
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            for conv in convos
        ]
        return {"text": texts}

    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=dataset.column_names,  # Keep only text column
    )

    # 4) Truncate to the number of samples needed
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Dataset prepared with {len(dataset)} samples")
    return dataset, tokenizer


def initialize_model():
    """
    Initialize Llama-3.1-8B model with LoRA adapters
    
    Returns:
        model: Model with LoRA adapters
        tokenizer: Tokenizer for the model
    """
    print("Initializing Llama-3.1-8B model...")
    
    # Load base model and tokenizer
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/Llama-3.1-8B",
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
        finetune_vision_layers=False,  # Llama is text-only
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
    
    return trainer


def train_with_profiling(trainer, profiling_mode="none", model_name="llama3_1_8b"):
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
    
    start_time = time.time()
    print(f"START_PROFILE: {start_time}")
    
    if profiling_mode == "proton":
        # Use Triton Proton profiler
        session_id = proton.start(name=f"llama3_1_8b_{model_name}", context="shadow")
        trainer.train()
        proton.finalize(session_id)
        print(f"Proton profile saved as llama3_1_8b_{model_name}.hatchet")
        
    elif profiling_mode == "torch":
        # Use PyTorch profiler
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            trainer.train()
        
        trace_file = f"llama3_1_8b_trace_{model_name}.json"
        prof.export_chrome_trace(trace_file)
        print(f"PyTorch profile saved as {trace_file}")
        
    else:
        # Standard training without profiling
        trainer.train()
    
    end_time = time.time()
    print(f"END_PROFILE: {end_time}")
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama-3.1-8B model using Unsloth and LoRA"
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
    
    args = parser.parse_args()
    
    print("="*60)
    print("Llama-3.1-8B Fine-tuning with Unsloth")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Profiling mode: {args.profiling}")
    print(f"  - Number of samples: {args.num_samples}")
    print(f"  - Max training steps: {args.max_steps}")
    print(f"  - Batch size: {args.batch_size}")
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
    # model.save_pretrained("./llama3_1_8b_finetuned")
    # tokenizer.save_pretrained("./llama3_1_8b_finetuned")


if __name__ == "__main__":
    main()