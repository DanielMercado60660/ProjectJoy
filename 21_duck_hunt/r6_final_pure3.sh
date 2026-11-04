#!/bin/sh
# Pure-PPO final recipe (frozen from screening on training courses), seeds 0-2 serially.
cd "$(dirname "$0")"
for s in 0 1 2; do
  sh r6_final_chainB.sh $s --reward time --hid 256 --gamma 0.99 --lr 3e-4 --lr_sched linear --update_device gpu
done
