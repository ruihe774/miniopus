#!/bin/sh
# Concatenates test/audio/*.wav into raw mono s16le PCM at 16 kHz (SILK) and 48 kHz (CELT).
set -e
cd "$(dirname "$0")/.."
mkdir -p target/bench
for sr in 16000 48000; do
  for f in test/audio/*.wav; do ffmpeg -v error -i "$f" -f s16le -ac 1 -ar $sr -; done > target/bench/pcm$sr.raw
  ls -l target/bench/pcm$sr.raw
done
