#!/usr/bin/env python3
"""
Standalone training script for Llama-3-70B-bnb-4bit model using Unsloth
This script demonstrates fine-tuning Llama-3-70B with 4-bit quantization and LoRA adapters
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

    # 2) Set Llama-3 chat template
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3")

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
    Initialize Llama-3-70B-bnb-4bit model with LoRA adapters
    Note: This model comes pre-quantized in 4-bit format
    
    Returns:
        model: Model with LoRA adapters
        tokenizer: Tokenizer for the model
    """
    print("Initializing Llama-3-70B-bnb-4bit model...")
    print("Note: This model is already 4-bit quantized for optimal memory usage")
    
    # Load base model and tokenizer
    # The model is already 4-bit quantized, so we don't need to set load_in_4bit
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/llama-3-70b-bnb-4bit",
        max_seq_length=2048,
        dtype=None,  # Auto-detect dtype
        load_in_4bit=True,  # This model is already 4-bit quantized
        # cache_dir="/scratch/jlee436/unsloth/model"
    )
    
    print("Adding LoRA adapters...")
    # Configure LoRA adapters for parameter-efficient fine-tuning
    # Use smaller rank for larger models to save memory
    model = FastModel.get_peft_model(
        model,
        r=16,  # LoRA rank (can be 8, 16, 32, 64, 128)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,  # LoRA scaling parameter
        lora_dropout=0,  # LoRA dropout (0 for no dropout)
        bias="none",  # Bias training strategy
        use_gradient_checkpointing="unsloth",  # For memory efficiency with large models
        random_state=3407,  # For reproducibility
        use_rslora=False,  # Rank stabilized LoRA
        loftq_config=None,  # LoftQ configuration
    )
    
    print(f"Model initialized with {model.num_parameters():,} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model, tokenizer


def create_trainer(model, tokenizer, train_dataset):
    """
    Create SFTTrainer with training configuration optimized for large models
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: The training dataset
    
    Returns:
        SFTTrainer instance
    """
    print("Creating trainer with SFT configuration...")
    print("Using memory-efficient settings for 70B model...")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,  # Add validation dataset here if needed
        args=SFTConfig(
            # Dataset configuration
            dataset_text_field="text",
            max_seq_length=2048,
            
            # Batch size and accumulation (reduced for 70B model)
            per_device_train_batch_size=1,  # Reduced batch size for memory
            gradient_accumulation_steps=8,  # Effective batch size = 1 * 8 = 8
            
            # Training steps
            warmup_steps=5,
            max_steps=30,  # Increase for longer training
            
            # Learning rate and optimizer
            learning_rate=2e-4,
            optim="adamw_8bit",  # 8-bit AdamW for memory efficiency
            weight_decay=0.01,
            lr_scheduler_type="linear",
            
            # Memory optimization
            gradient_checkpointing=True,  # Essential for 70B model
            fp16=not torch.cuda.is_bf16_supported(),  # Use fp16 if bf16 not supported
            bf16=torch.cuda.is_bf16_supported(),  # Use bf16 if available
            
            # Logging
            logging_steps=5,
            report_to="none",  # Change to "wandb" or "tensorboard" for logging
            
            # Other settings
            seed=3407,
            output_dir="./llama3_70b_outputs",
            
            # Additional memory optimizations
            optim_args="rank=64, update_proj_gap=200",  # Paged AdamW settings
            max_grad_norm=0.3,  # Gradient clipping
        ),
    )
    
    return trainer


def train_with_profiling(trainer, profiling_mode="none", model_name="llama3_70b_bnb_4bit"):
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
    
    # Memory usage before training
    if torch.cuda.is_available():
        print(f"GPU memory allocated before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"GPU memory reserved before training: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    start_time = time.time()
    print(f"START_PROFILE: {start_time}")
    
    if profiling_mode == "proton":
        # Use Triton Proton profiler
        session_id = proton.start(name=f"llama3_70b_{model_name}", context="shadow")
        trainer.train()
        proton.finalize(session_id)
        print(f"Proton profile saved as llama3_70b_{model_name}.hatchet")
        
    elif profiling_mode == "torch":
        # Use PyTorch profiler with memory profiling
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            profile_memory=True,
            record_shapes=True
        ) as prof:
            trainer.train()
        
        trace_file = f"llama3_70b_trace_{model_name}.json"
        prof.export_chrome_trace(trace_file)
        print(f"PyTorch profile saved as {trace_file}")
        
        # Print memory usage summary
        print("\nTop 10 memory consuming operations:")
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        
    else:
        # Standard training without profiling
        trainer.train()
    
    end_time = time.time()
    print(f"END_PROFILE: {end_time}")
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    
    # Memory usage after training
    if torch.cuda.is_available():
        print(f"GPU memory allocated after training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"GPU memory reserved after training: {torch.cuda.memory_reserved()/1024**3:.2f} GB")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama-3-70B-bnb-4bit model using Unsloth and LoRA"
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
        default=100,  # Reduced default for 70B model
        help="Number of training samples to use (default: 100)"
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
        default=1,
        help="Per-device training batch size (default: 1 for 70B model)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16, can be 8, 16, 32, 64, 128)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Llama-3-70B-bnb-4bit Fine-tuning with Unsloth")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Model: unsloth/llama-3-70b-bnb-4bit (pre-quantized)")
    print(f"  - Profiling mode: {args.profiling}")
    print(f"  - Number of samples: {args.num_samples}")
    print(f"  - Max training steps: {args.max_steps}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  - LoRA rank: {args.lora_r}")
    print(f"  - Max sequence length: {args.max_seq_length}")
    print("="*60 + "\n")
    
    # Check GPU availability and memory
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
        print(f"CUDA version: {torch.version.cuda}")
        print("="*60 + "\n")
    else:
        print("WARNING: No GPU detected. This model requires GPU for training.")
        return
    
    # Initialize model and tokenizer
    model, tokenizer = initialize_model()
    
    # Prepare dataset
    train_dataset, tokenizer = prepare_dataset(tokenizer, args.num_samples)
    
    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset)
    
    # Override settings if specified
    if args.max_steps != 30:
        trainer.args.max_steps = args.max_steps
    if args.batch_size != 1:
        trainer.args.per_device_train_batch_size = args.batch_size
    if args.gradient_accumulation_steps != 8:
        trainer.args.gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # Start training with profiling
    train_with_profiling(trainer, args.profiling, "standalone")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    # Optional: Save the model (LoRA adapters only)
    # print("Saving LoRA adapters...")
    # model.save_pretrained("./llama3_70b_lora_adapters")
    # tokenizer.save_pretrained("./llama3_70b_lora_adapters")
    
    # Optional: Merge and save full model (requires lots of disk space)
    # print("Merging and saving full model...")
    # model.save_pretrained_merged("./llama3_70b_merged", tokenizer, save_method="merged_16bit")


if __name__ == "__main__":
    main()