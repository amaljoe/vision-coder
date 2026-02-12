#!/bin/bash
#SBATCH --job-name=dgx-job
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=6-00:00:00
#SBATCH --ntasks-per-node=256

# Print node info
hostname
nvidia-smi

source scripts/setup-compute.sh

# --------- SOCKS PROXY VIA LOGIN NODE ---------
LOGIN_NODE=login1
SOCKS_PORT=1080

echo "Starting SOCKS proxy via $LOGIN_NODE on port $SOCKS_PORT"

ssh -N -D ${SOCKS_PORT} ${LOGIN_NODE} &
SSH_PID=$!

# Give tunnel time to come up
sleep 5

# ------ SOCKS PROXY VIA LOGIN NODE ---------


/usr/sbin/sshd -f ~/.ssh/sshd_4422/sshd_config

# Activate vcoder environment
mamba activate /dev/shm/vcoder3

# Launch TensorBoard for training monitoring
export PLAYWRIGHT_BROWSERS_PATH=/dev/shm/pw-browsers
tensorboard --logdir=~/workspace/vision-coder/notebooks/Qwen3-VL-2B-HTMLCSS/runs --host=0.0.0.0 --port=6006 &

sleep infinity

