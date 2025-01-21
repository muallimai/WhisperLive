# Source code
git clone https://github.com/muallimai/WhisperLive.git
cd WhisperLive/

# venv
sudo apt install python3.10-venv
python3.10 -m venv .venv
source .venv/bin/activate

# Libraries
pip install -r requirements/server.txt
pip install ctranslate2==4.4.0

# Download turbo v3
pip install transformers[torch]>=4.23
mkdir models
ct2-transformers-converter --model openai/whisper-large-v3-turbo --output_dir models/whisper-large-v3-turbo-ct2 --copy_files tokenizer.json preprocessor_config.json

python run_server.py -fw models/whisper-large-v3-turbo-ct2