import torch 
from torch import nn

class ONE_TO_ONE_FIX_MODEL(nn.Module):
  def __init__(self,n_dim=len(c620_op_col)+len(c660_op_col)+len(c670_op_col)):
    super(ONE_TO_ONE_FIX_MODEL,self).__init__()
    self.n_dim = n_dim
    for i in range(n_dim):
      setattr(self,'O_net{}'.format(i+1),self._build_O_net(1,1))
    self.apply(self._init_weights)
  
  @staticmethod
  def _init_weights(m):
    if hasattr(m,'weight'):
      torch.nn.init.xavier_uniform(m.weight)
    if hasattr(m,'bias'):
      m.bias.data.fill_(0)

  def _build_O_net(self,input_shape=1,output_shape=1):
    return nn.Linear(input_shape,output_shape)
            
  def forward(self,x):
    output = torch.tensor([])
    for i in range(self.n_dim):
      O_net = getattr(self,'O_net{}'.format(i+1))
      value = O_net(x[:,[i]])
      output = torch.cat((output,value),dim=1)
    return output
