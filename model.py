import torch
from torch import nn
from sp2wt import sp2wt

# 1 C620_OP 
class C620_OP(nn.Module):
    def __init__(self,c_dim=3,f_dim=41,o_dim=9,h_dim=128):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.o_dim = o_dim
        self.h_dim = h_dim
        self.net = nn.Sequential(
            nn.Linear(self.c_dim+self.f_dim,self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim,self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim,self.o_dim),
        )
    
    def forward(self,c,f):
        return self.net(torch.cat((c,f),dim =-1))

# 2 C620_SF
class C620_SF(nn.Module):
    def __init__(self,c_dim=3,f_dim=41,o_dim=41*4,h_dim=256):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.o_dim = o_dim
        self.h_dim = h_dim
        self.net = nn.Sequential(
            nn.Linear(self.c_dim+self.f_dim,self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim,self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim,self.o_dim),
            nn.Sigmoid(), # range[0,1]
        )
    
    def forward(self,c,f):
        sp = self.net(torch.cat((c,f),dim =-1))
        return sp[:,:41], sp[:,41:41*2], sp[:,41*2:41*3], sp[:,41*3:]

# 3 TR_C620
class TR_C620(nn.Module):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def sp2wt(x,s):
        a = 100*x*s
        b = torch.diag(x@s.T).reshape(-1,1)
        b = torch.clamp(b,1e-8,1e+8)
        return a/b
    
    def forward(self,x,s1,s2,s3,s4):
        w1 = self.sp2wt(x,s1)
        w2 = self.sp2wt(x,s2)
        w3 = self.sp2wt(x,s3)
        w4 = self.sp2wt(x,s4)
        return w1,w2,w3,w4

# 4 C660_MF
class C660_MF(nn.Module):
    def __init__(self,f_dim=41,w_dim=41,h_dim=128,o_dim=2):
        super().__init__()
        self.f_dim = f_dim
        self.w_dim = w_dim
        self.h_dim = h_dim
        self.o_dim = o_dim
        self.net = nn.Sequential(
            nn.Linear(self.f_dim + self.w_dim,self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim,self.o_dim),
            nn.Sigmoid(),# range [0,1]
            )
    
    def forward(self,f,w):
        y = self.net(torch.cat((f,w),dim=-1))
        return y[:,[0]],y[:,[1]]

# 5 TR_C620_T651
class TR_C620_T651(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, c620_x,t651_x,c620_mf,t651_mf):
        mix = c620_x * c620_mf + t651_x * t651_mf
        return  mix

# 6 C660_OP
class C660_OP(nn.Module):
    def __init__(self,c_dim=2,f_dim=41,o_dim=9,h_dim=128):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.o_dim = o_dim
        self.h_dim = h_dim
        self.net = nn.Sequential(
            nn.Linear(self.c_dim+self.f_dim,self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim,self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim,self.o_dim),
        )

    def forward(self,c,f):
        return self.net(torch.cat((c,f),dim =-1))

# 7 C660_SF
class C660_SF(nn.Module):
    def __init__(self,c_dim=2,f_dim=41,o_dim=41*4,h_dim=256):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.o_dim = o_dim
        self.h_dim = h_dim
        self.net = nn.Sequential(
            nn.Linear(self.c_dim+self.f_dim,self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim,self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim,self.o_dim),
            nn.Sigmoid(),# range[0,1]
        )
    
    def forward(self,c,f):
        sp = self.net(torch.cat((c,f),dim =-1))
        return sp[:,:41], sp[:,41:41*2], sp[:,41*2:41*3], sp[:,41*3:]

# 8 TR_C660
class TR_C660(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x,s1,s2,s3,s4):
        w1,w2,w3,w4 = sp2wt(x,s1),sp2wt(x,s2),sp2wt(x,s3),sp2wt(x,s4)
        return w1,w2,w3,w4

# 9 C670_SF
class C670_SF(nn.Module):
    def __init__(self,f_dim = 82,o_dim=82,h_dim=128):
        super().__init__()
        self.f_dim = f_dim
        self.o_dim = o_dim
        self.h_dim = h_dim
        self.net = nn.Sequential(
            nn.Linear(self.f_dim,self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim,self.o_dim),
            )
    
    def forward(self,c620_w4,c660_w4):
        x = torch.cat((c620_w4,c660_w4),dim=-1)
        s = self.net(x)
        s1,s2 = s[:,:41],s[:,41:]
        return s1,s2

# 10 C670 OP
class C670_OP(nn.Module):
    def __init__(self,f_dim=82,o_dim=7,h_dim=128):
        super().__init__()
        self.f_dim = f_dim
        self.o_dim = o_dim 
        self.h_dim = h_dim
        self.net = nn.Sequential(
            nn.Linear(self.f_dim,self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim,self.o_dim),
        )
    
    def forward(self,c620_w4,c660_w4):
        x = torch.cat((c620_w4,c660_w4),dim=-1)
        op = self.net(x)
        return op

# 11 TR_C670
class TR_C670(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x,s1,s2):
        w1,w2 = sp2wt(x,s1),sp2wt(x,s2)
        return w1,w2