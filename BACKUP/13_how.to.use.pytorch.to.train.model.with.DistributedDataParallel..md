# [how to use pytorch to train model with DistributedDataParallel ](https://github.com/AlexiFeng/gitblog/issues/13)

Many nouns are also used in course "Parallel Computing"
```python
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1)
FLAGS = parser.parse_args()
local_rank = int(FLAGS.local_rank)

# 新增3：DDP backend初始化
#   a.根据local_rank来设定当前使用哪块GPU
torch.cuda.set_device(local_rank)
#   b.初始化DDP，使用默认backend(nccl)就行。如果是CPU模型运行，需要选择其他后端。
dist.init_process_group(backend='nccl')
device = torch.device("cuda", local_rank)
model=SimpleNet().to(device) #init model
model = DDP(model, device_ids=[local_rank], output_device=local_rank)  #use DDP
```
and then,must use distributedsampler,distribute different data to each process.
```python
train_sampler  = torch.utils.data.distributed.DistributedSampler(train_dataset)
```
use one process to save model
```python
if dist.get_rank()==0:
    meg.save(model.module, save_path+str(cur_epoch)+ ".pth")
```

then should use barrier?I've no idea.