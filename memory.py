# 报错如下
"""
****** Conduct Training ******
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
CUDA_VISIBLE_DEVICES: [0]
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "D:\Ana3\envs\pytorch2\lib\multiprocessing\spawn.py", line 116, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "D:\Ana3\envs\pytorch2\lib\multiprocessing\spawn.py", line 125, in _main
    prepare(preparation_data)
  File "D:\Ana3\envs\pytorch2\lib\multiprocessing\spawn.py", line 236, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
  File "D:\Ana3\envs\pytorch2\lib\multiprocessing\spawn.py", line 287, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
  File "D:\Ana3\envs\pytorch2\lib\runpy.py", line 288, in run_path
    return _run_module_code(code, init_globals, run_name,
  File "D:\Ana3\envs\pytorch2\lib\runpy.py", line 97, in _run_module_code
    _run_code(code, mod_globals, init_globals,
  File "D:\Ana3\envs\pytorch2\lib\runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "D:\gra_project\ABSA-QUAD-master\main.py", line 303, in <module>
    trainer.fit(model)
  File "D:\Ana3\envs\pytorch2\lib\site-packages\pytorch_lightning\trainer\trainer.py", line 918, in fit
    self.single_gpu_train(model)
  File "D:\Ana3\envs\pytorch2\lib\site-packages\pytorch_lightning\trainer\distrib_parts.py", line 163, in single_gpu_train
    model.cuda(self.root_gpu)
  File "D:\Ana3\envs\pytorch2\lib\site-packages\torch\nn\modules\module.py", line 552, in _apply
    param_applied = fn(param)
  File "D:\Ana3\envs\pytorch2\lib\site-packages\torch\nn\modules\module.py", line 637, in <lambda>
    return self._apply(lambda t: t.cuda(device))
RuntimeError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 2.00 GiB total capacity; 418.20 MiB already allocated; 18.06 MiB free; 458.00 MiB reserved
 in total by PyTorch)
"""

import torch

dtype = torch.float
N, D_in, H, D_out = 64, 1000, 100, 10

device = torch.device("cuda")
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
gpu_memory_log()
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    #print(t, loss.item())
    loss.backward()
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()

print(loss.item())
gpu_memory_log()
