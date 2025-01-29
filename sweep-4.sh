echo "Starting sweep over beta and temperature hyperparameters"

echo "Run 2: test-1b"
accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
    --output_dir Qwen2.5-1.5B-GRPO \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name AI-MO/NuminaMath-TIR \
    --max_prompt_length 256 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --logging_steps 1 \
    --wandb_run_name test-1b \
    --bf16

echo "Waiting 30 seconds before next run..."
sleep 30

echo "Run 3: 1b-beta-low (beta=0.004)"
accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
    --output_dir Qwen2.5-1.5B-GRPO \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name AI-MO/NuminaMath-TIR \
    --max_prompt_length 256 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --logging_steps 1 \
    --wandb_run_name 1b-beta-low \
    --beta 0.004 \
    --bf16

echo "Waiting 30 seconds before next run..."
sleep 30

echo "Run 4: 1b-beta-high (beta=0.4)"
accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
    --output_dir Qwen2.5-1.5B-GRPO \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name AI-MO/NuminaMath-TIR \
    --max_prompt_length 256 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --logging_steps 1 \
    --wandb_run_name 1b-beta-high \
    --beta 0.4 \
    --bf16

echo "Sweep completed!"