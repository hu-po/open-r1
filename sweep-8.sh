echo "Starting sweep over dataset size and learning rate hyperparameters"

echo "Run 1: 7b-2kdata-2e5"
accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
    --output_dir DeepSeek-R1-Distill-Qwen-7B-GRPO \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset_name AI-MO/NuminaMath-TIR \
    --max_prompt_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 1 \
    --wandb_run_name 7b-2kdata-2e5 \
    --max_train_samples 2000 \
    --learning_rate 2e-5 \
    --bf16

echo "Waiting 30 seconds before next run..."
sleep 30

# echo "Run 2: 7b-2kdata-2e5-beta4"
# accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
#     --output_dir DeepSeek-R1-Distill-Qwen-7B-GRPO \
#     --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#     --dataset_name AI-MO/NuminaMath-TIR \
#     --max_prompt_length 256 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --logging_steps 1 \
#     --wandb_run_name 7b-2kdata-2e5-beta4 \
#     --max_train_samples 2000 \
#     --learning_rate 2e-5 \
#     --beta 0.4 \
#     --bf16

# echo "Waiting 30 seconds before next run..."
# sleep 30

# echo "Run 3: 7b-2kdata-2e5-beta004"
# accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
#     --output_dir DeepSeek-R1-Distill-Qwen-7B-GRPO \
#     --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#     --dataset_name AI-MO/NuminaMath-TIR \
#     --max_prompt_length 256 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --logging_steps 1 \
#     --wandb_run_name 7b-2kdata-2e5-beta004 \
#     --max_train_samples 2000 \
#     --learning_rate 2e-5 \
#     --beta 0.004 \
#     --bf16

# echo "Waiting 30 seconds before next run..."
# sleep 30

# echo "Run 4: 7b-2kdata-2e5-temp099"
# accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
#     --output_dir DeepSeek-R1-Distill-Qwen-7B-GRPO \
#     --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#     --dataset_name AI-MO/NuminaMath-TIR \
#     --max_prompt_length 256 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --logging_steps 1 \
#     --wandb_run_name 7b-2kdata-2e5-temp099 \
#     --max_train_samples 2000 \
#     --learning_rate 2e-5 \
#     --temperature 0.99 \
#     --bf16

# echo "Waiting 30 seconds before next run..."
# sleep 30

# echo "Run 5: 7b-2kdata-2e5-temp07"
# accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
#     --output_dir DeepSeek-R1-Distill-Qwen-7B-GRPO \
#     --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#     --dataset_name AI-MO/NuminaMath-TIR \
#     --max_prompt_length 256 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --logging_steps 1 \
#     --wandb_run_name 7b-2kdata-2e5-temp07 \
#     --max_train_samples 2000 \
#     --learning_rate 2e-5 \
#     --temperature 0.7 \
#     --bf16

# echo "Sweep completed!"