# export WANDB_MODE=offline
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export HF_EVALUATE_OFFLINE=1

cd ~/workspace/ixbrl-tagging
# source .venv/bin/activate

# Export proxy envs
export SOCKS_PORT=1080
export ALL_PROXY="socks5h://localhost:${SOCKS_PORT}"
export http_proxy=$ALL_PROXY
export https_proxy=$ALL_PROXY
export HTTP_PROXY=$ALL_PROXY
export HTTPS_PROXY=$ALL_PROXY

# pip install pysocks socksio httpx[socks]


