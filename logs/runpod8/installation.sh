git clone https://github.com/orangingq/torchtitan # 비번은 카톡에
cd torchtitan 

# 온갖 설치
pip install torchtitan
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
pip install -r requirements.txt
apt-get update # 무시 가능
apt-get install -y pciutils # 무시 가능
apt-get install rsync tmux 
#! pipelining/stage 에다가 pipeline_logger 넣기

# wandb login
wandb login # 키는 wandb 사이트에서

# debug하려면 vscode에서 debug extension 깔고 launch.json 설정 & ANSI code

# server 4에서 tokenizer 보내기
scp -P 13957 -i ~/.ssh/id_ed25519 -r assets/tokenizer root@64.247.196.118:/workspace/torchtitan/assets/

# Llama 3.1 8B 모델 다운로드
huggingface-cli login # 토큰은 server4/skipp에
huggingface-cli download meta-llama/Llama-3.1-8B --include "original/*" --local-dir /workspace/torchtitan_data/base_model/Llama-3.1-8B
huggingface-cli download meta-llama/Llama-3.2-1B --include "original/*" --local-dir /data2/shcho/torchtitan/base_model/Llama-3.2-1B


# DCP 포맷으로 변환
python ./scripts/checkpoint_conversion/convert_from_llama.py /workspace/torchtitan_data/base_model/Llama-3.1-8B/original /workspace/torchtitan_data/base_model/Llama-3.1-8B/original_dcp

# 실제 돌릴 때는
nohup bash logs/runpod8/0922_main/run.sh > logs/runpod8/0922_main/nohup.ans 2>&1 &
# tmux
bash logs/runpod8/0922_main/run.sh

# 학습 중 로그를 terminal에서 보고 싶을 때:
tail -f logs/runpod8/0922_main/nohup.ans

# 끌 때는
ps -ef | grep runpod8 | grep .sh     # bash file 전체의 pid 확인
ps -ef | grep timelyfreeze.train    # 해당 run의 pid 확인
kill -9 <pid>

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

# evaluation
# huggingface의 safetensor format으로 변형해도, sharded 형태로 저장되어 있어서 index file을 따로 만들어야 함.
# for example...
nano /data2/shcho/torchtitan/checkpoint/1020_gpipe_nofreeze_dm4/step-500/sharded/model.safetensors.index.json
{
  "metadata": {
    "total_size": 961583888
  },
  "weight_map": {
    "model.layers.0.weight": "shard-00001-model-00001-of-00002.safetensors",
    "model.layers.1.weight": "shard-00001-model-00002-of-00002.safetensors"
  }
}