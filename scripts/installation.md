

# Installation
### Step 1
```bash
git clone https://github.com/orangingq/torchtitan # 비번은 카톡에
cd torchtitan 
conda create -n llm python=3.13
conda activate llm
# pip3 install torch torchvision # Please install PyTorch before proceeding.
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
pip3 install -r requirements.txt
apt-get update # 무시 가능
apt-get install -y pciutils # 무시 가능
```

### Step 2: Manual 설정
1. pipelining/stage 에다가 pipeline_logger 넣기
2. VSCode Extension: debug하려면 vscode에서 debug extension 깔고 launch.json 설정 & ANSI code

### Step 3: Wandb Login
```bash
wandb login # 키는 wandb 사이트에서
```

### Step 4: Tokenizer Download

```bash
hf download meta-llama/Llama-3.2-1B-Instruct --include "original/*" "tokenizer*.json" "special_tokens_map.json" --local-dir /home/shcho/torchtitan/assets/tokenizer/Llama-3.2-1B-Instruct
hf download meta-llama/Llama-3.1-8B-Instruct --include "original/*" "tokenizer*.json" "special_tokens_map.json" --local-dir /opt/dlami/nvme/DMLAB/shcho/torchtitan/assets/tokenizer/Llama-3.1-8B-Instruct
hf download meta-llama/Llama-3.1-8B --include "original/*" "tokenizer*.json" "special_tokens_map.json" --local-dir /opt/dlami/nvme/DMLAB/shcho/torchtitan/assets/tokenizer/Llama-3.1-8B
```

### Step 5: Pretrained Llama Model Download & Format Conversion
```bash
huggingface-cli login # 토큰은 server4/skipp에
```
```bash
hf download meta-llama/Llama-3.2-1B-Instruct --include "original/*" "model.safetensors" --local-dir /data2/shcho/torchtitan/base_model/Llama-3.2-1B-Instruct
hf download meta-llama/Llama-3.2-3B-Instruct --include "original/*" "model.safetensors" --local-dir /data2/shcho/torchtitan/base_model/Llama-3.2-3B-Instruct
# H200 server
hf download meta-llama/Llama-3.1-8B-Instruct --include "original/*" "*.safetensors" "model.safetensors.index.json" "config.json" --local-dir /opt/dlami/nvme/DMLAB/shcho/torchtitan_data/base_model/Llama-3.1-8B-Instruct
hf download meta-llama/Llama-3.1-8B --include "original/*" "*.safetensors" "model.safetensors.index.json" "config.json" --local-dir /opt/dlami/nvme/DMLAB/shcho/torchtitan_data/base_model/Llama-3.1-8B
```

#### Format Change: Huggingface -> DCP Format
```bash
python ./scripts/checkpoint_conversion/convert_from_llama.py /workspace/torchtitan_data/base_model/Llama-3.1-8B/original /workspace/torchtitan_data/base_model/Llama-3.1-8B/original_dcp
python ./scripts/checkpoint_conversion/convert_from_llama.py /data2/shcho/torchtitan/base_model/Llama-3.2-1B/original /data2/shcho/torchtitan/base_model/Llama-3.2-1B/original_dcp
python ./scripts/checkpoint_conversion/convert_from_llama.py /opt/dlami/nvme/DMLAB/shcho/torchtitan_data/base_model/Llama-3.1-8B-Instruct/original /opt/dlami/nvme/DMLAB/shcho/torchtitan_data/base_model/Llama-3.1-8B-Instruct/original_dcp
python ./scripts/checkpoint_conversion/convert_from_llama.py /opt/dlami/nvme/DMLAB/shcho/torchtitan_data/base_model/Llama-3.1-8B/original /opt/dlami/nvme/DMLAB/shcho/torchtitan_data/base_model/Llama-3.1-8B/original_dcp
```

# Out-of-date installation steps
```bash
# server 4에서: checkpoint 받아오기
# scp: 느린 버전
# scp -P 49353 -i ~/.ssh/id_ed25519 -r root@64.247.196.118:/workspace/torchtitan_data/base_model/Llama* /data2/shcho/torchtitan/checkpoint # 느린 버전
# scp -P 49353 -i ~/.ssh/id_ed25519 -r root@64.247.196.118:/workspace/torchtitan_data/checkpoint/* /data2/shcho/torchtitan/checkpoint # 느린 버전
# runpod에서 삭제하기
rm -rf /workspace/torchtitan_data/checkpoint/*
# # rsync: 빠른 버전
# conda install -c conda-forge rsync
rsync -avz --progress \
   -e "ssh -P 13957 -i ~/.ssh/id_ed25519" \
   root@64.247.196.118:/workspace/base_model/Llama*/ \
   /data2/shcho/torchtitan/checkpoint/
rsync -avz --partial --progress \
    -e "ssh -p 13957 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no" \
    root@64.247.196.118:/workspace/torchtitan_data/checkpoint/* \
    /data2/shcho/torchtitan/checkpoint/

# convert DCP -> HF
torchrun --nproc_per_node=1 --nnodes=1 --standalone --role=rank --tee=3 -m scripts.checkpoint_conversion.convert_to_hf /workspace/torchtitan_data/checkpoint/0922_gpipe_apf_dm4/step-50 /workspace/torchtitan_data/checkpoint/0922_gpipe_apf_dm4/hf_step-50 --model_name=llama3 --model_flavor=8B
# convert base model DCP -> HF
torchrun --nproc_per_node=1 --nnodes=1 --standalone --local_addr=127.0.0.1 --role=rank --tee=3 -m scripts.checkpoint_conversion.convert_to_hf /data2/shcho/torchtitan/checkpoint/base_model/Llama-3.2-1B/original_dcp /data2/shcho/torchtitan/checkpoint/base_model/Llama-3.2-1B/original_dcp --model_name=llama3 --model_flavor=1B
```