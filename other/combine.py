from model import C620_OP,C620_SF,TR_C620,C660_MF,TR_C620_T651,C660_OP,C660_SF,TR_C660,C670_OP,C670_SF,TR_C670
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.c620_op = C620_OP()
        self.c620_sf = C620_SF()
        self.tr_c620 = TR_C620()
        self.c660_mf = C660_MF()
        self.tr_c620_t651 = TR_C620_T651()
        self.c660_op = C660_OP()
        self.c660_sf = C660_SF()
        self.tr_c660 = TR_C660()
        self.c670_op = C670_OP()
        self.c670_sf = C670_SF()
        self.tr_c670 = TR_C670()

    def forward(self,c620_case,c620_feed,t651_feed,c660_case):
        # 620
        c620_op = self.c620_op(c620_case,c620_feed)
        c620_s1, c620_s2, c620_s3, c620_s4 = self.c620_sf(c620_case, c620_feed)
        c620_w1, c620_w2, c620_w3, c620_w4 = self.tr_c620(c620_feed, c620_s1, c620_s2, c620_s3, c620_s4)
        
        # 651
        m_c620,m_t651 = self.c660_mf(t651_feed, c620_w3)
        print(c620_w3.shape,t651_feed.shape,m_c620.shape,m_t651.shape)
        c660_feed = self.tr_c620_t651(c620_w3,t651_feed,m_c620,m_t651)
        
        # 660
        c660_op = self.c660_op(c660_case,c660_feed)
        c660_s1, c660_s2 ,c660_s3 ,c660_s4 = self.c660_sf(c660_case,c660_feed)
        c660_w1, c660_w2, c660_w3 ,c660_w4 = self.tr_c660(c660_feed, c660_s1, c660_s2, c660_s3, c660_s4)

        # 670
        c670_op = self.c670_op(c620_w4,c660_w4)
        c670_s1, c670_s2 = self.c670_sf(c620_w4,c660_w4)
        c670_w1 , c670_w2 = self.tr_c670(c620_w4+c660_w4,c670_s1,c670_s2)
        
        return c620_op, c620_w1, c620_w2, c620_w3, c620_w4 , \
               c660_op, c660_w1, c660_w2, c660_w3, c660_w4 , \
               c670_op, c670_w1, c670_w2    


if __name__ == '__main__':
    # fake data
    c620_case = torch.rand(64,3)
    c660_case = torch.rand(64,2)
    feed_c620 = torch.rand(64,41)
    feed_t651 = torch.rand(64,41)

    # forward test
    model = Model()
    x = [c620_case,feed_c620,feed_t651,c660_case]
    y = model(*x)
    print('can forward')

    # backward test
    loss = 0 
    for p in y:
        loss += p.mean() 
    loss.backward()
    print('can backward')

    # writer
    writer = SummaryWriter()
    writer.add_graph(model,x)
    print('can tensorboard')
    writer.close()