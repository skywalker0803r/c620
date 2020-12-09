import torch

def sp2wt(x,s):
    a = 100*x*s
    b = torch.diag(x@s.T).reshape(-1,1)
    b = torch.clamp(b,1e-8,1e+8)
    return a/b