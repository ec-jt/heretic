export NODE_RANK=<0|1|2|3>
export OMP_NUM_THREADS=8
export JOBLIB_TEMP_FOLDER=/tmp
export TMPDIR=/tmp

torchrun \
  --nnodes=4 \
  --nproc-per-node=8 \
  --node-rank=$NODE_RANK \
  --master-addr=10.101.8.160 \
  --master-port=29500 \
  -m heretic.main \
  --model /mnt/nvme0/models/moonshotai/Kimi-K2.5 \
  --distributed \
  --distributed-backend nccl \
  --tensor-parallel \
  --tensor-parallel-plan auto \
  --cuda-alloc-conf expandable_segments:True
