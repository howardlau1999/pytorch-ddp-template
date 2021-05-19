#!/bin/bash
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --nnodes=${SLURM_JOB_NUM_NODES} \
  --node_rank=${SLURM_NODEID} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  ddp.py
