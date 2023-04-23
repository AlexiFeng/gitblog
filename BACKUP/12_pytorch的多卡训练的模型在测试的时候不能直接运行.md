# [pytorch的多卡训练的模型在测试的时候不能直接运行](https://github.com/AlexiFeng/gitblog/issues/12)

这个坑遇到过两次了，多卡联合训练的时候模型直接存储会多一个module。很多时候用dataparallel测试不太现实。
解决办法1：
```python
# save model
if num_gpu ==  1:
    torch.save(model.module.state_dict(), 'net.pth')
 else:
    torch.save(model.state_dict(),  'net.pth')
```
办法2：
把训练好的模型里的model字符删除(我目前用的主要是这种）反正也不麻烦。
```python
pth = torch.load('./626.pth')
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in pth.items():
    name =  k[7:] # remove  'module'
    new_state_dict[name]=v
model.load_state_dict(new_state_dict)
model.eval()
```
> 原文链接：https://blog.csdn.net/szn1316159505/article/details/129225188