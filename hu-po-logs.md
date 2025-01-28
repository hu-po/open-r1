# 2025.01.27

created fork of open-r1

https://github.com/hu-po/open-r1

connecting to lambdalabs instance (8xH100 @ ~20$/hr)

https://docs.lambdalabs.com/public-cloud/on-demand/connecting-instance/

Generate a new key and download to local machine

https://cloud.lambdalabs.com/ssh-keys

```bash
mv ~/Downloads/open-r1-key.pem ~/dev/open-r1/
chmod 600 /home/oop/dev/open-r1/lambdalabs-ssh-key.pem
ssh -i /home/oop/dev/open-r1/lambdalabs-ssh-key.pem ubuntu@192.222.52.162
```

```bash
ubuntu@192-222-52-162:~$ nvidia-smi
Tue Jan 28 15:24:59 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:61:00.0 Off |                    0 |
| N/A   31C    P0             70W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA H100 80GB HBM3          On  |   00000000:62:00.0 Off |                    0 |
| N/A   29C    P0             69W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA H100 80GB HBM3          On  |   00000000:63:00.0 Off |                    0 |
| N/A   27C    P0             68W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA H100 80GB HBM3          On  |   00000000:64:00.0 Off |                    0 |
| N/A   31C    P0             72W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA H100 80GB HBM3          On  |   00000000:6A:00.0 Off |                    0 |
| N/A   35C    P0             72W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA H100 80GB HBM3          On  |   00000000:6B:00.0 Off |                    0 |
| N/A   29C    P0             69W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA H100 80GB HBM3          On  |   00000000:6C:00.0 Off |                    0 |
| N/A   31C    P0             70W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA H100 80GB HBM3          On  |   00000000:6D:00.0 Off |                    0 |
| N/A   27C    P0             70W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

environment setup

```bash
python --version
sudo apt install git-lfs
git-lfs --version
cd open-r1-day2/ # name of the folder aka "filesystem" on the instance
git clone https://github.com/hu-po/open-r1.git
cd open-r1/ # navigate to cloned repo folder
pip install -e .
pip install wandb
wandb login # click on login link, paste in token
huggingface-cli login # click invalidate&refresh, paste in token
pip install latex2sympy2_extended # for GRPO
pip install math_verify # for GRPO
pip install tf-keras # for GRPO
pip install jinja2==3.1.0 # for GRPO
```

testing out GRPO

https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
https://huggingface.co/datasets/AI-MO/NuminaMath-TIR

```bash
accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
    --output_dir DeepSeek-R1-Distill-Qwen-7B-GRPO \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset_name AI-MO/NuminaMath-TIR \
    --max_prompt_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 10 \
    --bf16
```

visualizations on wandb

https://wandb.ai/hug/huggingface