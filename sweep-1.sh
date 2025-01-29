echo "Starting sweep over beta and temperature hyperparameters"

echo "Run 1: beta-smol (beta=0.004)"
accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
    --output_dir DeepSeek-R1-Distill-Qwen-7B-GRPO \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset_name AI-MO/NuminaMath-TIR \
    --max_prompt_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 1 \
    --wandb_run_name beta-smol \
    --beta 0.004 \
    --bf16

echo "Waiting 30 seconds before next run..."
sleep 30

echo "Run 2: beta-big (beta=0.4)"
accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
    --output_dir DeepSeek-R1-Distill-Qwen-7B-GRPO \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset_name AI-MO/NuminaMath-TIR \
    --max_prompt_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 1 \
    --wandb_run_name beta-big \
    --beta 0.4 \
    --bf16

echo "Waiting 30 seconds before next run..."
sleep 30

echo "Run 3: temp-big (temperature=1.5)"
accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
    --output_dir DeepSeek-R1-Distill-Qwen-7B-GRPO \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset_name AI-MO/NuminaMath-TIR \
    --max_prompt_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 1 \
    --wandb_run_name temp-big \
    --temperature 1.5 \
    --bf16

echo "Waiting 30 seconds before next run..."
sleep 30

echo "Run 4: temp-smol (temperature=0.2)"
accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
    --output_dir DeepSeek-R1-Distill-Qwen-7B-GRPO \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset_name AI-MO/NuminaMath-TIR \
    --max_prompt_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 1 \
    --wandb_run_name temp-smol \
    --temperature 0.2 \
    --bf16

echo "Sweep completed!"