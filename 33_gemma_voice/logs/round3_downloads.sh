#!/bin/bash
# Round-3 data: LibriTTS-R train-clean-360 + train-other-500 (openslr 141, CC BY 4.0) and Jenny (HF reach-vb/jenny_tts_dataset).
# The two LibriTTS-R files download in parallel from two OpenSLR mirrors (resumable, curl -C -), are md5-checked against
# openslr's md5sum.txt, extracted next to clean-100, then the archive is removed. Jenny parquets stay as downloaded.
set -u
cd /Users/danielmercado/ProjectJoy/data/libritts_r
get() {  # name md5 base_url
  f=$1.tar.gz
  for try in 1 2 3 4 5; do curl -sSL -C - -o download/$f $3/resources/141/$f && break; echo "$f retry $try"; sleep 30; done
  got=$(md5 -q download/$f); echo "$f md5 $got expected $2"
  [ "$got" = "$2" ] || { echo "MD5_MISMATCH $f"; return; }
  mkdir -p extracted && tar -xzf download/$f -C extracted && echo "$f extracted" && rm download/$f
}
get train_clean_360 382eb3e64394b3da6a559f864339b22c https://www.openslr.org &
get train_other_500 a37a8e9f4fe79d20601639bf23d1add8 https://openslr.trmal.net &
cd /Users/danielmercado/ProjectJoy
HF_HUB_DISABLE_XET=1 .venv/bin/python -c "
from huggingface_hub import snapshot_download
p=snapshot_download('reach-vb/jenny_tts_dataset', repo_type='dataset', local_dir='data/jenny', allow_patterns=['data/*.parquet','README.md'])
print('jenny at', p)"
wait
du -sh data/libritts_r/extracted/LibriTTS_R/* data/jenny
echo DOWNLOADS_DONE
