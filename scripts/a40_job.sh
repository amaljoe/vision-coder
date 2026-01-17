#!/bin/bash
#SBATCH --job-name=a40-job
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=64

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

# jupyter lab --ip=0.0.0.0 --port=8888

sleep infinity

