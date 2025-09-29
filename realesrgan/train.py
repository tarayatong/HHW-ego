# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline

import realesrgan.archs
import realesrgan.data
import realesrgan.models
import os
import torch
import torch.distributed as dist

# 获取 local_rank
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

# os.environ.setdefault('RANK', '0')
# os.environ.setdefault('WORLD_SIZE', '1')
# os.environ.setdefault('LOCAL_RANK', '0')


# 初始化分布式训练
#dist.init_process_group(backend="nccl")
if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
