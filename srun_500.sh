#!/bin/bash

export NCCL_DEBUG=info
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL



while
  port=$(shuf -n 1 -i 49152-65535)
  netstat -atun | grep -q "$port"
do
  continue
done

echo "$port"
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=${port} metricL.py\
 --train_size 500 \
 --batch_size_metricL 125 \
 --bank_size 1000 \
 --augpos_scale 0.04 \
 --augpos_prob 0.5 \
 --augpos_threshold 0.4 \
 --extra_prefix '' \
 --use_bn_embed \
 --alter_tau \
 --l96 \
 # --load_saved_metric \
