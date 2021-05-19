#/bin/sh
TIMESTAMP=$(date --rfc-3339="seconds" | tr " :" "_")
GPUS=0,1,2,3
NGPU_PER_NODE=4
NNODE=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=9315
MAIN_SCRIPT=ddp.py
SCRIPT_ARGS=
CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --nproc_per_node=$NGPU_PER_NODE --nnodes=$NNODE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $MAIN_SCRIPT $SCRIPT_ARGS
