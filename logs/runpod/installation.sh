pip install torchtitan

pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall

git clone https://github.com/orangingq/torchtitan
# 비번은 카톡에

cd torchtitan
pip install -r requirements.txt

# server 4에서
scp -P 41112 -i ~/.ssh/id_ed25519 -r assets/tokenizer root@38.128.232.57:/workspace/torchtitan/assets/

# debug하려면 vscode에서 debug extension 깔고 launch.json 설정


# 실제 돌릴 때는
nohup bash logs/runpod/0922_main/run.sh > logs/runpod/0922_main/nohup.out 2>&1 &