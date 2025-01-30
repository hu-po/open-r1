echo "Starting sweep over dataset size and learning rate hyperparameters"

echo "Run 1: 7b-2kdata-2e5-vllm"
accelerate launch --config_file configs/zero3.yaml --num_processes=7 src/open_r1/grpo.py \
    --output_dir DeepSeek-R1-Distill-Qwen-7B-GRPO \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset_name AI-MO/NuminaMath-TIR \
    --max_prompt_length 512 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 1 \
    --wandb_run_name 7b-2kdata-2e5-vllm \
    --max_train_samples 2000 \
    --learning_rate 2e-5 \
    --bf16 \
    --use_vllm \
    --vllm_device auto \
    --vllm_gpu_memory_utilization 0.7

echo "Waiting 30 seconds before next run..."
sleep 30
