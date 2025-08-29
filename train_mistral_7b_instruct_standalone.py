#!/usr/bin/env python3
"""
Standalone training script for Mistral-7B-Instruct model using Unsloth
This script demonstrates fine-tuning Mistral-7B-Instruct with LoRA adapters
"""

import os
import time
import argparse
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from trl import SFTTrainer, SFTConfig
import triton.profiler as proton


def prepare_dataset(tokenizer, num_samples=1000):
    print("Loading FineTome-100k dataset...")
    dataset = load_dataset("mlabonne/FineTome-100k", split="train")

    # 1) Standardize ShareGPT -> role/content
    dataset = standardize_sharegpt(dataset)  # No need to pass tokenizer

    # 2) Set Mistral chat template
    tokenizer = get_chat_template(tokenizer, chat_template="mistral")

    # 3) Apply template to each conversation
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
    Initialize Mistral-7B-Instruct model with LoRA adapters
    
    Returns:
        model: Model with LoRA adapters
        tokenizer: Tokenizer for the model
    """
    print("Initializing Mistral-7B-Instruct model...")
    
    # Load base model and tokenizer
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/mistral-7b-instruct-v0.3",
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
        r=8,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=8,  # LoRA scaling parameter
        lora_dropout=0,  # LoRA dropout
        bias="none",  # Bias training strategy
        use_gradient_checkpointing="unsloth",  # For memory efficiency
        random_state=3407,  # For reproducibility
        use_rslora=False,  # Rank stabilized LoRA
        loftq_config=None,  # LoftQ configuration
    )
    
    print(f"Model initialized with {model.num_parameters():,} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
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
            max_seq_length=2048,
            
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
            
            # Memory optimization
            gradient_checkpointing=True,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            
            # Logging
            logging_steps=15,
            report_to="none",  # Change to "wandb" or "tensorboard" for logging
            
            # Other settings
            seed=3407,
            output_dir="./mistral_7b_outputs",
            
            # Additional optimizations
            max_grad_norm=0.3,
        ),
    )
    
    # Apply response-only training for Mistral format
    trainer = train_on_responses_only(
        trainer,
        instruction_part="[INST]",
        response_part="[/INST]",
    )
    
    return trainer


def train_with_profiling(trainer, profiling_mode="none", model_name="mistral_7b_instruct"):
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
        session_id = proton.start(name=f"mistral_7b_{model_name}", context="shadow")
        trainer.train()
        proton.finalize(session_id)
        print(f"Proton profile saved as mistral_7b_{model_name}.hatchet")
        
    elif profiling_mode == "torch":
        # Use PyTorch profiler
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            profile_memory=True,
            record_shapes=True
        ) as prof:
            trainer.train()
        
        trace_file = f"mistral_7b_trace_{model_name}.json"
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
        description="Fine-tune Mistral-7B-Instruct model using Unsloth and LoRA"
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
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (default: 8)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
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
    print("Mistral-7B-Instruct Fine-tuning with Unsloth")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Model: unsloth/mistral-7b-instruct-v0.3")
    print(f"  - Profiling mode: {args.profiling}")
    print(f"  - Number of samples: {args.num_samples}")
    print(f"  - Max training steps: {args.max_steps}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  - LoRA rank: {args.lora_r}")
    print(f"  - Max sequence length: {args.max_seq_length}")
    print(f"  - 4-bit quantization: {args.load_in_4bit}")
    print(f"  - 8-bit quantization: {args.load_in_8bit}")
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
    
    # Update model loading parameters if specified
    if args.load_in_4bit or args.load_in_8bit:
        print("Re-initializing model with quantization settings...")
        model, tokenizer = FastModel.from_pretrained(
            model_name="unsloth/mistral-7b-instruct-v0.3",
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            full_finetuning=False,
        )
        
        # Re-apply LoRA
        model = FastModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_r,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    
    # Prepare dataset
    train_dataset, tokenizer = prepare_dataset(tokenizer, args.num_samples)
    
    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset)
    
    # Override settings if specified
    if args.max_steps != 30:
        trainer.args.max_steps = args.max_steps
    if args.batch_size != 2:
        trainer.args.per_device_train_batch_size = args.batch_size
    if args.gradient_accumulation_steps != 4:
        trainer.args.gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # Start training with profiling
    train_with_profiling(trainer, args.profiling, "standalone")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    # Optional: Save the model (LoRA adapters only)
    # print("Saving LoRA adapters...")
    # model.save_pretrained("./mistral_7b_lora_adapters")
    # tokenizer.save_pretrained("./mistral_7b_lora_adapters")
    
    # Optional: Merge and save full model
    # print("Merging and saving full model...")
    # model.save_pretrained_merged("./mistral_7b_merged", tokenizer, save_method="merged_16bit")


if __name__ == "__main__":
    main()