import transformers
print(transformers.__version__)  # 4.3.0

import sentencepiece
print(sentencepiece.__version__)  # 0.2.0

import pytorch_lightning
print(pytorch_lightning.__version__)  # 0.8.1

import torch
print(torch.__version__)  # 1.9.0+cu111
if torch.cuda.is_available():
    print("CUDA is available")  # CUDA is available
else:
    print("CUDA is not available")
