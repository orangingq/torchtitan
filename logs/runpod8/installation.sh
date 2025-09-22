git clone https://github.com/orangingq/torchtitan # 비번은 카톡에
cd torchtitan 

# 온갖 설치
pip install torchtitan
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
pip install -r requirements.txt
apt-get update # 무시 가능
apt-get install -y pciutils # 무시 가능

# server 4에서 tokenizer 보내기
scp -P 41112 -i ~/.ssh/id_ed25519 -r assets/tokenizer root@38.128.232.57:/workspace/torchtitan/assets/

# debug하려면 vscode에서 debug extension 깔고 launch.json 설정

# Llama 3.1 8B 모델 다운로드
huggingface-cli login # 토큰은 server4/skipp에
huggingface-cli download meta-llama/Llama-3.1-8B --include "original/*" --local-dir /workspace/torchtitan_data/base_model/Llama-3.1-8B

# DCP 포맷으로 변환
python ./scripts/checkpoint_conversion/convert_from_llama.py /workspace/torchtitan_data/base_model/Llama-3.1-8B/original /workspace/torchtitan_data/base_model/Llama-3.1-8B/original_dcp

# wandb login
wandb login # 키는 wandb 사이트에서

# 실제 돌릴 때는
nohup bash logs/runpod8/0922_main/run.sh > logs/runpod8/0922_main/nohup.ans 2>&1 &

# 학습 중 로그를 terminal에서 보고 싶을 때:
tail -f logs/runpod8/0922_main/nohup.ans

# 끌 때는
ps -ef | grep runpod8 | grep .sh     # bash file 전체의 pid 확인
ps -ef | grep timelyfreeze.train    # 해당 run의 pid 확인
kill -9 <pid>