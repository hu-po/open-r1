echo "Starting sweep over dataset size and learning rate hyperparameters"

echo "Run 2: small-dataset-high-lr (samples=1000, lr=1e-4)"
accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
    --output_dir DeepSeek-R1-Distill-Qwen-7B-GRPO \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset_name AI-MO/NuminaMath-TIR \
    --max_prompt_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 1 \
    --wandb_run_name small-dataset-high-lr \
    --max_train_samples 1000 \
    --learning_rate 1e-4 \
    --bf16

echo "Waiting 30 seconds before next run..."
sleep 30

echo "Run 3: 7b-10kdata-1e5 (samples=2000, lr=1e-5)"
accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
    --output_dir DeepSeek-R1-Distill-Qwen-7B-GRPO \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset_name AI-MO/NuminaMath-TIR \
    --max_prompt_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 1 \
    --wandb_run_name 7b-10kdata-1e5 \
    --max_train_samples 2000 \
    --learning_rate 1e-5 \
    --bf16

echo "Waiting 30 seconds before next run..."
sleep 30

echo "Run 4: 7b-2kdata-1e4 (samples=2000, lr=1e-4)"
accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
    --output_dir DeepSeek-R1-Distill-Qwen-7B-GRPO \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset_name AI-MO/NuminaMath-TIR \
    --max_prompt_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 1 \
    --wandb_run_name 7b-2kdata-1e4 \
    --max_train_samples 2000 \
    --learning_rate 1e-4 \
    --bf16

echo "Sweep completed!"