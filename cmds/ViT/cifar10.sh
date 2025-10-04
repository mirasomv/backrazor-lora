save_dir="."

devices="0"
port=7296
n_gpu=1

#devices="0,1,2,3"
#port=7256
#n_gpu=4

lr=0.01


# === Setup power logging ===
log_dir="logs"
mkdir -p $log_dir
power_log="$log_dir/lora8-16.csv"
echo "timestamp,power.draw [W]" > "$power_log"

# === Start power monitoring ===
monitor_power() {
    while true; do
        timestamp=$(date +%s)
	power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -n 1)
	echo "$timestamp,$power" >> "$power_log"
        sleep 1
    done
}

monitor_power &
monitor_pid=$!

# === Start training + timing ===
start_time=$(date +%s)

lr=0.01

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch \
  --nproc_per_node=1 --master_port 12345 \
  ViT/train.py \
  --name cifar10-lora-r8-a16-drop0.1-80000 \
  --learning_rate 3e-4 --num_workers 6 --output_dir ./outputs \
  --dataset cifar10 --model_type ViT-S_16 \
  --pretrained_dir ./pretrain/ViT-S_16.npz \
  --train_batch_size 32 --eval_batch_size 32 \
  --num_steps 15640 --eval_every 1564 \
  --train_lora_only \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1


end_time=$(date +%s)
elapsed=$((end_time - start_time))

# === Stop power logging ===
kill $monitor_pid
echo "✅ Training completed in $elapsed seconds."

# === Setup power logging ===
log_dir="logs"
mkdir -p $log_dir
power_log="$log_dir/lora4-8.csv"
echo "timestamp,power.draw [W]" > "$power_log"

# === Start power monitoring ===
monitor_power() {
    while true; do
        timestamp=$(date +%s)
	power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -n 1)
	echo "$timestamp,$power" >> "$power_log"
        sleep 1
    done
}

monitor_power &
monitor_pid=$!

# === Start training + timing ===
start_time=$(date +%s)

lr=0.01

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch \
  --nproc_per_node=1 --master_port 12345 \
  ViT/train.py \
  --name cifar10-lora-r8-a16-drop0.1-80000 \
  --learning_rate 3e-4 --num_workers 6 --output_dir ./outputs \
  --dataset cifar10 --model_type ViT-S_16 \
  --pretrained_dir ./pretrain/ViT-S_16.npz \
  --train_batch_size 32 --eval_batch_size 32 \
  --num_steps 15640 --eval_every 1564 \
  --train_lora_only \
  --lora_rank 4 \
  --lora_alpha 8 \
  --lora_dropout 0.1


end_time=$(date +%s)
elapsed=$((end_time - start_time))

# === Stop power logging ===
kill $monitor_pid
echo "✅ Training completed in $elapsed seconds."

# === Setup power logging ===
log_dir="logs"
mkdir -p $log_dir
power_log="$log_dir/lora8-16_CIFAR100.csv"
echo "timestamp,power.draw [W]" > "$power_log"

# === Start power monitoring ===
monitor_power() {
    while true; do
        timestamp=$(date +%s)
	power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -n 1)
	echo "$timestamp,$power" >> "$power_log"
        sleep 1
    done
}

monitor_power &
monitor_pid=$!

# === Start training + timing ===
start_time=$(date +%s)

lr=0.01

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch \
  --nproc_per_node=1 --master_port 12345 \
  ViT/train.py \
  --name cifar100-lora-r8-a16-drop0.1-80000 \
  --learning_rate 3e-4 --num_workers 6 --output_dir ./outputs \
  --dataset cifar100 --model_type ViT-S_16 \
  --pretrained_dir ./pretrain/ViT-S_16.npz \
  --train_batch_size 32 --eval_batch_size 32 \
  --num_steps 15640 --eval_every 1564 \
  --train_lora_only \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1


end_time=$(date +%s)
elapsed=$((end_time - start_time))

# === Stop power logging ===
kill $monitor_pid
echo "✅ Training completed in $elapsed seconds."

# === Setup power logging ===
log_dir="logs"
mkdir -p $log_dir
power_log="$log_dir/lora4-8_CIFAR100.csv"
echo "timestamp,power.draw [W]" > "$power_log"

# === Start power monitoring ===
monitor_power() {
    while true; do
        timestamp=$(date +%s)
	power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -n 1)
	echo "$timestamp,$power" >> "$power_log"
        sleep 1
    done
}

monitor_power &
monitor_pid=$!

# === Start training + timing ===
start_time=$(date +%s)

lr=0.01

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch \
  --nproc_per_node=1 --master_port 12345 \
  ViT/train.py \
  --name cifar100-lora-r8-a16-drop0.1-80000 \
  --learning_rate 3e-4 --num_workers 6 --output_dir ./outputs \
  --dataset cifar100 --model_type ViT-S_16 \
  --pretrained_dir ./pretrain/ViT-S_16.npz \
  --train_batch_size 32 --eval_batch_size 32 \
  --num_steps 15640 --eval_every 1564 \
  --train_lora_only \
  --lora_rank 4 \
  --lora_alpha 8 \
  --lora_dropout 0.1


end_time=$(date +%s)
elapsed=$((end_time - start_time))

# === Stop power logging ===
kill $monitor_pid
echo "✅ Training completed in $elapsed seconds."


# === Setup power logging ===
log_dir="logs"
mkdir -p $log_dir
power_log="$log_dir/lora8-16_pets.csv"
echo "timestamp,power.draw [W]" > "$power_log"

# === Start power monitoring ===
monitor_power() {
    while true; do
        timestamp=$(date +%s)
	power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -n 1)
	echo "$timestamp,$power" >> "$power_log"
        sleep 1
    done
}

monitor_power &
monitor_pid=$!

# === Start training + timing ===
start_time=$(date +%s)

lr=0.01

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch \
  --nproc_per_node=1 --master_port 12345 \
  ViT/train.py \
  --name Pet37-lora-r8-a16-drop0.1-80000 \
  --learning_rate 3e-4 --num_workers 6 --output_dir ./outputs \
  --dataset Pet37 --model_type ViT-S_16 \
  --pretrained_dir ./pretrain/ViT-S_16.npz \
  --train_batch_size 32 --eval_batch_size 32 \
  --num_steps 15640 --eval_every 1564 \
  --train_lora_only \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1


end_time=$(date +%s)
elapsed=$((end_time - start_time))

# === Stop power logging ===
kill $monitor_pid
echo "✅ Training completed in $elapsed seconds."

# === Setup power logging ===
log_dir="logs"
mkdir -p $log_dir
power_log="$log_dir/lora4-8_pets.csv"
echo "timestamp,power.draw [W]" > "$power_log"

# === Start power monitoring ===
monitor_power() {
    while true; do
        timestamp=$(date +%s)
	power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -n 1)
	echo "$timestamp,$power" >> "$power_log"
        sleep 1
    done
}

monitor_power &
monitor_pid=$!

# === Start training + timing ===
start_time=$(date +%s)

lr=0.01

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch \
  --nproc_per_node=1 --master_port 12345 \
  ViT/train.py \
  --name Pet37-lora-r8-a16-drop0.1-80000 \
  --learning_rate 3e-4 --num_workers 6 --output_dir ./outputs \
  --dataset Pet37 --model_type ViT-S_16 \
  --pretrained_dir ./pretrain/ViT-S_16.npz \
  --train_batch_size 32 --eval_batch_size 32 \
  --num_steps 15640 --eval_every 1564 \
  --train_lora_only \
  --lora_rank 4 \
  --lora_alpha 8 \
  --lora_dropout 0.1


end_time=$(date +%s)
elapsed=$((end_time - start_time))

# === Stop power logging ===
kill $monitor_pid
echo "✅ Training completed in $elapsed seconds."

