# -*- coding = utf-8 -*-
# @Time : 2022/1/16 19:15
# @Author : SBP
# @File : seed.py
# @Software : PyCharm

import os
import torch
import numpy as np
import random

seed = 1234
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
