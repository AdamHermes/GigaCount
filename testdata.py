import torch
nprocs = torch.cuda.device_count()
print(nprocs)