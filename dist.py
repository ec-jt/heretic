export OMP_NUM_THREADS=8
export HERETIC_DISTRIBUTED_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO torchrun --nnodes=4 --nproc-per-node=8 --node-rank=0 --master-addr=10.101.8.160 --master-port=29500 -m heretic.main --model /mnt/nvme0/models/moonshotai/Kimi-K2.5 --distributed --distributed-backend nccl --tensor-parallel --tensor-parallel-plan auto --cuda-alloc-conf expandable_segments:True
