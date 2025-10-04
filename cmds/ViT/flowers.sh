save_dir="."

devices="0"
port=7260
n_gpu=1

backPruneRatio=0.8
lr=1e-3

# === Setup power logging ===
log_dir="logs"
mkdir -p $log_dir
power_log="$log_dir/flowers_br80_2300.csv"
log_out="$log_dir/flowers_br80_2300.log"
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

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
ViT/train.py --name flowers-lr${lr}-BackRazor0.8 --learning_rate ${lr} --num_workers 6 --output_dir ${save_dir} \
--dataset flowers --model_type ViT-S_16 --pretrained_dir ${save_dir}/pretrain/ViT-S_16.npz \
--new_backrazor --back_prune_ratio 0.8 \
--train_batch_size 32 --eval_batch_size 32 \
--num_steps 1564 --eval_every 156 \
2>&1 | tee "$log_out"

end_time=$(date +%s)
elapsed=$((end_time - start_time))

# === Stop power logging ===
kill $monitor_pid
echo "✅ Training completed in $elapsed seconds." | tee -a "$log_out"


# === Setup power logging ===
log_dir="logs"
mkdir -p $log_dir
power_log="$log_dir/flowers_ftfull2300.csv"
log_out="$log_dir/flowers_ftfull2300.log"
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

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
ViT/train.py --name flowers-lr${lr}-FTFULLNEW --learning_rate ${lr} --num_workers 6 --output_dir ${save_dir} \
--dataset flowers --model_type ViT-S_16 --pretrained_dir ${save_dir}/pretrain/ViT-S_16.npz \
--train_batch_size 32 --eval_batch_size 32 \
--num_steps 1564 --eval_every 156 \
2>&1 | tee "$log_out"

end_time=$(date +%s)
elapsed=$((end_time - start_time))

# === Stop power logging ===
kill $monitor_pid
echo "✅ Training completed in $elapsed seconds." | tee -a "$log_out"


backPruneRatio=0.95
lr=1e-3

# === Setup power logging ===
log_dir="logs"
mkdir -p $log_dir
power_log="$log_dir/flowers_br95_2300.csv"
log_out="$log_dir/flowers_br95_2300.log"
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

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
ViT/train.py --name flowers-lr${lr}-B128-BackRazor${backPruneRatio} --learning_rate ${lr} --num_workers 6 --output_dir ${save_dir} \
--dataset flowers --model_type ViT-S_16 --pretrained_dir ${save_dir}/pretrain/ViT-S_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} \
--train_batch_size 32 --eval_batch_size 32 \
--num_steps 1564 --eval_every 156 \
2>&1 | tee "$log_out"

end_time=$(date +%s)
elapsed=$((end_time - start_time))

# === Stop power logging ===
kill $monitor_pid
echo "✅ Training completed in $elapsed seconds." | tee -a "$log_out"

backPruneRatio=0.9
lr=1e-3

# === Setup power logging ===
log_dir="logs"
mkdir -p $log_dir
power_log="$log_dir/flowers_br90_2300.csv"
log_out="$log_dir/flowers_br90_2300.log"
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

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
ViT/train.py --name flowers-lr${lr}-B128-BackRazor${backPruneRatio} --learning_rate ${lr} --num_workers 6 --output_dir ${save_dir} \
--dataset flowers --model_type ViT-S_16 --pretrained_dir ${save_dir}/pretrain/ViT-S_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} \
--train_batch_size 32 --eval_batch_size 32 \
--num_steps 1564 --eval_every 156 \
2>&1 | tee "$log_out"

end_time=$(date +%s)
elapsed=$((end_time - start_time))

# === Stop power logging ===
kill $monitor_pid
echo "✅ Training completed in $elapsed seconds." | tee -a "$log_out"


# === Setup power logging ===
log_dir="logs"
mkdir -p $log_dir
power_log="$log_dir/flowers_bitfit2300.csv"
log_out="$log_dir/flowers_bitfit2300.log"
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

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
ViT/train.py --name flowers-lr${lr}-bitfit --learning_rate ${lr} --num_workers 6 --output_dir ${save_dir} \
--dataset flowers --model_type ViT-S_16 --pretrained_dir ${save_dir}/pretrain/ViT-S_16.npz \
--bitfit \
--train_batch_size 32 --eval_batch_size 32 \
--num_steps 1564 --eval_every 156 \
2>&1 | tee "$log_out"

end_time=$(date +%s)
elapsed=$((end_time - start_time))

# === Stop power logging ===
kill $monitor_pid
echo "✅ Training completed in $elapsed seconds." | tee -a "$log_out"

# === Setup power logging ===
log_dir="logs"
mkdir -p $log_dir
power_log="$log_dir/flowers_ftfull230022.csv"
log_out="$log_dir/flowers_ftfull230022.log"
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

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
ViT/train.py --name flowers-lr${lr}-FTFULLNEW230022 --learning_rate ${lr} --num_workers 6 --output_dir ${save_dir} \
--dataset flowers --model_type ViT-S_16 --pretrained_dir ${save_dir}/pretrain/ViT-S_16.npz \
--train_batch_size 32 --eval_batch_size 32 \
--num_steps 15640 --eval_every 1564 \
2>&1 | tee "$log_out"

end_time=$(date +%s)
elapsed=$((end_time - start_time))

# === Stop power logging ===
kill $monitor_pid
echo "✅ Training completed in $elapsed seconds." | tee -a "$log_out"



