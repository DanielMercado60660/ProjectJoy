#!/bin/bash
# Finish train_other_500.tar.gz: the rest of the file in 4 byte ranges from two mirrors (elda, trmal) in parallel,
# each part resumable; then append the parts in order, md5 check, extract next to clean-100/360, remove the archive.
set -u
cd /Users/danielmercado/ProjectJoy/data/libritts_r
F=download/train_other_500.tar.gz; MD5=a37a8e9f4fe79d20601639bf23d1add8; TOTAL=46843185428
S0=$(stat -f %z $F); REM=$((TOTAL - S0)); N=4; CH=$(( (REM + N - 1) / N ))
MIRRORS=(https://openslr.elda.org https://openslr.trmal.net https://openslr.elda.org https://openslr.trmal.net)
echo "have $S0, remaining $REM in $N parts of $CH"
part() {  # k
  k=$1; a=$((S0 + k * CH)); b=$((a + CH - 1)); [ $b -ge $TOTAL ] && b=$((TOTAL - 1)); want=$((b - a + 1)); p=download/part$k
  touch $p
  for try in $(seq 1 30); do
    have=$(stat -f %z $p); [ $have -ge $want ] && break
    curl -fsSL -m 1800 -r $((a + have))-$b ${MIRRORS[$k]}/resources/141/train_other_500.tar.gz >> $p
    sleep 5
  done
  echo "part$k $(stat -f %z $p) / $want"
}
for k in 0 1 2 3; do part $k & done
wait
for k in 0 1 2 3; do cat download/part$k >> $F && rm download/part$k; done
got=$(md5 -q $F); echo "md5 $got expected $MD5"
[ "$got" = "$MD5" ] || { echo MD5_MISMATCH; exit 1; }
tar -xzf $F -C extracted && echo "train_other_500 extracted" && rm $F
du -sh extracted/LibriTTS_R/*
echo OTHER500_DONE
