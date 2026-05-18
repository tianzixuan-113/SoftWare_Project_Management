import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import csv
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def visualize_model_results(results_list, labels_list, save_path=None):
    """
    可视化模型结果与标签的函数
    
    参数:
    results_list (list): 包含n个[batch_size, m]大小的结果张量的列表
    labels_list (list): 包含n个[batch_size, m]大小的标签张量的列表
    save_path (str, optional): 图表保存路径，默认为None(不保存)
    """
    # 检查输入是否有效
    if len(results_list) != len(labels_list):
        raise ValueError("结果列表和标签列表的长度必须相同")
    
    # 在batch_size维度上合并所有结果和标签
    merged_results = torch.cat(results_list, dim=0)  # [batch_size*n, m]
    merged_labels = torch.cat(labels_list, dim=0)    # [batch_size*n, m]
    
    # 获取维度信息
    batch_size_times_n, m = merged_results.shape
    
    # 创建一个包含m个子图的图形
    fig, axes = plt.subplots(m, 1, figsize=(10, 4 * m))
    if m == 1:
        axes = [axes]  # 确保axes始终是列表
    
    # 为每个元素创建折线图
    for i in range(m):
        ax = axes[i]
        # 绘制结果(红色)和标签(蓝色)
        ax.plot(range(batch_size_times_n), merged_results[:, i].cpu().numpy(), 'r-', label='结果')
        ax.plot(range(batch_size_times_n), merged_labels[:, i].cpu().numpy(), 'b-', label='标签')
        
        # 设置图表标题和标签
        ax.set_title(f'元素 {i+1} 的结果与标签对比')
        ax.set_xlabel('样本索引')
        ax.set_ylabel('值')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    # 如果指定了保存路径，则保存图表
    if save_path:
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")
    
    plt.show()

class StockDataset(Dataset):
    def __init__(self,data_path,size,datalength,id,more,b_days=1,labelkind=0):
        self.b_days=b_days
        self.labelkind=labelkind
        self.more=more
        self.label_base=torch.tensor([[1/(3**0.5),1/(3**0.5),1/(3**0.5)],
                             [1/(3**0.5),-1/(3**0.5),1/(3**0.5)],
                             [-1/(3**0.5),1/(3**0.5),1/(3**0.5)],
                             [-1/(3**0.5),-1/(3**0.5),1/(3**0.5)],
                             [1/(3**0.5),1/(3**0.5),-1/(3**0.5)],
                             [1/(3**0.5),-1/(3**0.5),-1/(3**0.5)],
                             [-1/(3**0.5),1/(3**0.5),-1/(3**0.5)],
                             [-1/(3**0.5),-1/(3**0.5),-1/(3**0.5)]])
        self.data=[]
        self.label=[]
        self.size=size
        self.len=size
        self.rarelabels=[]
        self.datalength=datalength
        self.Mm1 = StandardScaler()
        for path in data_path:
            df = pd.read_csv(f"{path}/data/_{id}.csv",encoding="utf-8")
            df = np.array(df,dtype='float32')#将pandas读取的数据转化为array
            # x_data=torch.from_numpy(df)
            self.data.append(df)
            label_str=f'{path}/label/_{id}.csv'
            df = pd.read_csv(label_str,encoding="utf-8")
            df = np.array(df,dtype='float32')#将pandas读取的数据转化为array
            # y_data=torch.from_numpy(df)
            self.label.append(df)
        
    def __getitem__(self, index):
        batch=index//self.size#得到批次数
        number=index%self.size#得到该批次的索引
        if(not self.more):
            # a=torch.tensor([self.data[batch][number:number+self.datalength,0]]).squeeze()
            # b=torch.from_numpy(self.Mm1.fit_transform(self.data[batch][number:number+self.datalength,1:]))
            # c=torch.from_numpy(self.label[batch][number:number+self.datalength,0:8])
            # d=torch.from_numpy(self.label[batch][number+self.datalength,0:8])
            # e=torch.tensor([self.data[batch][number+self.datalength-1,0]]).squeeze()
            # f=torch.from_numpy(self.label[batch][number+self.datalength-1,0:8])
            return torch.tensor([self.data[batch][number:number+self.datalength+1,0]]).squeeze(),torch.from_numpy(self.data[batch][number:number+self.datalength+1,1:]),torch.from_numpy(self.label[batch][number:number+self.datalength+1,self.labelkind])
        
    def __len__(self):
        return self.len
    
def sinusoidal_position_embedding(date,batch_size, nums_head, max_len, output_dim,device):
    # (max_len, 1)
    #date=torch.tensor([math.sin(x.item()*2/(365*math.pi)) for x in date])
    date=date.view(batch_size,max_len,-1)
    date=date.repeat(1,1,output_dim//2)
    theta=torch.Tensor([2*math.pi/365]).repeat(batch_size,max_len,output_dim//2)

    date=date*theta
      
    emb = torch.stack([torch.sin(date), torch.cos(date)], dim=-1)
    emb=emb.reshape(batch_size,max_len,output_dim)

    # (bs, head, max_len, output_dim//2, 2)
    embeddings = emb.repeat(nums_head,1,1,1)  # 在bs维度重复，其他维度都是1不重复
    embeddings=embeddings.transpose(0,1)

    return embeddings


# %%

def RoPE(date,q, k):
    # q,k: (bs, head, max_len, output_dim)
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[-1]

    # (bs, head, max_len, output_dim)
    pos_emb = sinusoidal_position_embedding(date,batch_size, nums_head, max_len, output_dim, q.device)


    # cos_pos,sin_pos: (bs, head, max_len, output_dim)
    # 看rope公式可知，相邻cos，sin之间是相同的，所以复制一遍。如(1,2,3)变成(1,1,2,2,3,3)
    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1)  # 将奇数列信息抽取出来也就是cos 拿出来并复制
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # 将偶数列信息抽取出来也就是sin 拿出来并复制

    # q,k: (bs, head, max_len, output_dim)
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)  # reshape后就是正负交替了



    # 更新qw, *对应位置相乘
    q = q * cos_pos + q2 * sin_pos

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    # 更新kw, *对应位置相乘
    k = k * cos_pos + k2 * sin_pos

    return q, k

def get_len_mask(b: int, max_len: int, feat_lens: torch.Tensor, device: torch.device) -> torch.Tensor:
    attn_mask = torch.ones((b, max_len, max_len), device=device)
    for i in range(b):
        attn_mask[i, :, :feat_lens[i]] = 0
    return attn_mask.to(torch.bool)

def get_subsequent_mask(b: int, max_len: int, device: torch.device) -> torch.Tensor:
    """
    Args:
        b: batch-size.
        max_len: the length of the whole seqeunce.
        device: cuda or cpu.
    """
    return torch.triu(torch.ones((b, max_len, max_len), device=device), diagonal=1).to(torch.bool)     # or .to(torch.uint8)

def get_enc_dec_mask(
    b: int, max_feat_len: int, feat_lens: torch.Tensor, max_label_len: int, device: torch.device
) -> torch.Tensor:
    attn_mask = torch.zeros((b, max_label_len, max_feat_len), device=device)       # (b, seq_q, seq_k)
    for i in range(b):
        attn_mask[i, :, feat_lens[i]:] = 1
    return attn_mask.to(torch.bool)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, num_heads, p=0.):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p)
        
        # linear projections
        self.W_Q = nn.Linear(d_model, d_k * num_heads)
        self.W_K = nn.Linear(d_model, d_k * num_heads)
        self.W_V = nn.Linear(d_model, d_v * num_heads)
        self.W_out = nn.Linear(d_v * num_heads, d_model)

        # Normalization
        # References: <<Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification>>
        nn.init.normal_(self.W_Q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_K.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_V.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.W_out.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

    def forward(self, Q, K, V, attn_mask,date, **kwargs):
        N = Q.size(0)
        q_len, k_len = Q.size(1), K.size(1)
        d_k, d_v = self.d_k, self.d_v
        num_heads = self.num_heads

        # multi_head split
        Q = self.W_Q(Q).view(N, -1, num_heads, d_k).transpose(1, 2)
        K = self.W_K(K).view(N, -1, num_heads, d_k).transpose(1, 2)
        V = self.W_V(V).view(N, -1, num_heads, d_v).transpose(1, 2)
        Q,K=RoPE(date,Q,K)
        
        # pre-process mask 
        if attn_mask is not None:
            assert attn_mask.size() == (N, q_len, k_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)    # broadcast
            attn_mask = attn_mask.bool()

        # calculate attention weight
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e4)
        attns = torch.softmax(scores, dim=-1)        # attention weights
        attns = self.dropout(attns)

        # calculate output
        output = torch.matmul(attns, V)

        # multi_head merge
        output = output.transpose(1, 2).contiguous().reshape(N, -1, d_v * num_heads)
        output = self.W_out(output)

        return output
    

class LabelCon(nn.Module):
    def __init__(self,size):
        super(LabelCon,self).__init__()
        self.l0=nn.Linear(size,16)
        self.l1=nn.Linear(16,64)
        self.l2=nn.Linear(64,256)
        self.l3=nn.Linear(256,512)
        self.l4=nn.Linear(512,128)
        self.l5=nn.Linear(128,32)
        self.l6=nn.Linear(32,8)
        
    def forward(self,x):
        x=F.relu(self.l0(x))
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=F.relu(self.l4(x))
        x=F.relu(self.l5(x))
        x=self.l6(x)
        return x

class LinerNN(nn.Module):
    def __init__(self,length,size,hiddensize,heads):
        super(LinerNN,self).__init__()
        hdim=hiddensize//heads
        self.length=length
        
        self.l=nn.Linear(size,hiddensize)
        
        self.attn1=MultiHeadAttention(hdim,hdim,size,heads)
        self.norm_at1 = nn.LayerNorm(size)
        self.l1=nn.Linear(length*size,2048)
        self.l2=nn.Linear(2048,1024)
        self.l3=nn.Linear(1024,512)
        self.l4=nn.Linear(512,256)
        self.l5=nn.Linear(256,128)
        self.l0=nn.Linear(128,size)
        
        self.attn2=MultiHeadAttention(hdim,hdim,size,heads)
        self.norm_at2 = nn.LayerNorm(size)
        self.l6=nn.Linear((length+1)*size,2048)
        self.l7=nn.Linear(2048,1024)
        self.l8=nn.Linear(1024,512)
        self.l9=nn.Linear(512,256)
        self.l10=nn.Linear(256,128)
        self.l00=nn.Linear(128,size)
        
    def forward(self,x,date):
        normal=x
        b=x.size(0)
        #enc_mask = get_len_mask(b, max_feat_len, X_lens, device)
        next=self.attn1(normal,normal,normal,None,date[:,0:self.length])
        normal=self.norm_at1(normal+next)
        
        normal=normal.view(b,-1)
        normal=F.relu(self.l1(normal))        
        normal=F.relu(self.l2(normal))        
        normal=F.relu(self.l3(normal))        
        normal=F.relu(self.l4(normal))
        normal=F.relu(self.l5(normal))
        normal=self.l0(normal)
        normal=normal.unsqueeze(1)
        
        normal=torch.cat([x,normal],dim=1)
        
        next=self.attn2(normal,normal,normal,None,date)
        normal=self.norm_at2(normal+next)
        
        normal=normal.view(b,-1)
        normal=F.relu(self.l6(normal))
        normal=F.relu(self.l7(normal))
        normal=F.relu(self.l8(normal))
        normal=F.relu(self.l9(normal))
        normal=F.relu(self.l10(normal))
        normal=self.l00(normal)
        normal=normal.unsqueeze(1)
        
        normal = torch.cat([x,normal],dim=1)
        normal=normal[:,:,0:4]
        
        return normal
 
def labelwrite(start,high,low,close):
    if(start<=close and high<=close and start<=low):
        z=[0]    #low=start<end=high:1
    elif(start<=close and high>close and start<=low):
        z=[1]    #low=start<end<high:2
    elif(start<=close and high<=close and start>low):
        z=[2]     #low<start<end=high:3
    elif(start<=close and high>close and start>low):
        z=[3]      #low<start<end<high:4
    elif(start>close and high>start and close>low):
        z=[4]      #low=end<start=high:5
    elif(start>close and high<=start and close>low):
        z=[5]      #low=end<start<high:6
    elif(start>close and high>start and close<=low):
        z=[6]      #low<end<start=high:7
    elif(start>close and high<=start and close<=low):
        z=[7]      #low<end<start<high:8
    return z   

if __name__ == "__main__":
    torch.manual_seed(114514)

    # constants
    batch_size = 10                 # batch size
    max_feat_len = 22               # the maximum length of input sequence
    max_label_len = 22               # the maximum length of output sequence
    fbank_dim = 108                 # the dimension of input feature
    hidden_dim = 512                # the dimension of hidden layer
    vocab_size = 8                  # the size of vocabulary

    # dummy data
    # fbank_feature = torch.randn(batch_size, max_feat_len, fbank_dim)        # input sequence
    feat_lens = torch.full((batch_size,), max_feat_len)              # the length of each input sequence in the batch
    label = torch.zeros(batch_size, max_label_len)      # output sequence
    label_lens = torch.full((batch_size,),max_label_len)             # the length of each output sequence in the batch
    l0=['F:/stockPre/data/data_ComplexFormed_complexLabel/train']
    l1=['F:/stockPre/data/data_ComplexFormed_complexLabel/test']
    idx=0
    for xx in range(10):      
        
        model=LinerNN(21,108,512,8)
        model0=LabelCon(4)
        
        train_dataset=StockDataset(l0,4800,21,xx,False,b_days=1)
        train_loader=DataLoader(train_dataset,shuffle=True,batch_size=10)
        test_dataset=StockDataset(l1,1100,21,xx,False,b_days=1)
        test_loader=DataLoader(test_dataset,shuffle=True,batch_size=10)
            
        criterion = nn.MSELoss()
        criterion0 = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)      
        optimizer0 = optim.Adam(model0.parameters(), lr=0.0001)    
        #optimizer = optim.SGD(transformer.parameters(),lr=0.005,momentum=0.9)
        
        best=0
        for epoch in range(10):
            running_loss_data = 0.0
            running_loss_label = 0.0
            correct0 = 0
            relectcorrect0=0
            total0 = 0
            for batch_idx, data in enumerate(train_loader, 0):           
                date,inputs,label = data      
                label=label[:,-1,:]
                optimizer.zero_grad()
                input=inputs[:,0:21,:]
                outputs=model(input,date)
                outputs=outputs.squeeze()
                In=inputs[:,:,0:4]
                if(batch_idx%2==0):
                    loss = criterion(outputs, In)
                    loss.backward()
                    optimizer.step()
                    running_loss_data += loss.item()
                
                optimizer0.zero_grad()
                outputs=model0(outputs[:,-1,:])
                if(batch_idx%2==1):
                    loss0 = criterion0(outputs, label)
                    loss0.backward()
                    optimizer0.step()
                    running_loss_label += loss0.item()
                
                
                if batch_idx % 100 == 99:
                    print('[%d,%5d] loss_data: %.3f'% (epoch + 1, batch_idx + 1, running_loss_data / batch_idx))
                    print('[%d,%5d] loss_label: %.3f'% (epoch + 1, batch_idx + 1, running_loss_label / batch_idx))
                
                _,predicted = torch.max(outputs.data,dim=1)
                _,llabels=torch.max(label.data,dim=1)
                
                shape=torch.ones(batch_size)
                shape=shape.view(-1,1)
                predicted=predicted.view_as(shape)
                llabels=llabels.view_as(shape)
                total0 += label.size(0)
                correct0 +=(predicted == llabels).sum().item()
                for x,y in zip(predicted,llabels):
                        relectcorrect0+=(int(x.item()<4 and y.item()<4)+int(x.item()>3 and y.item()>3))
            print('[%d,%5d] loss_data: %.3f'% (epoch + 1, batch_idx + 1, running_loss_data / batch_idx))
            print('[%d,%5d] loss_label: %.3f'% (epoch + 1, batch_idx + 1, running_loss_label / batch_idx))
            print('Accuracy on test set: %d %%' % (100 * correct0 / total0))
            print('reAccuracy on test set: %d %%' % (100 * relectcorrect0 / total0))
            #     target=label[:,-1,:]
            #     out=outputs[:,-1,:]
            #     _,predicted = torch.max(out.data,dim=1)
            #     #l2_sim = compute_similarity(target, label_base, method='cosine')
            #     _,llabels=torch.max(target.data,dim=1)
                
            #     shape=torch.ones(batch_size)
            #     shape=shape.view(-1,1)
            #     predicted=predicted.view_as(shape)
            #     llabels=llabels.view_as(shape)
            #     total0 += target.size(0)
            #     correct0 +=(predicted == llabels).sum().item()
            #     for x,y in zip(predicted,llabels):
            #             relectcorrect0+=(int(x.item()<4 and y.item()<4)+int(x.item()>3 and y.item()>3))
            # print('[%d,%5d] loss: %.3f'% (epoch + 1, batch_idx + 1, running_loss / batch_idx))
            # print('Accuracy on test set: %d %%' % (100 * correct0 / total0))
            # print('reAccuracy on test set: %d %%' % (100 * relectcorrect0 / total0))
                    
            # correct = 0
            # relectcorrect=0
            # total = 0
            # with torch.no_grad():
            #     for data in test_loader:
            #         date,inputs,label = data
            #         input=inputs[:,0:21,:]
            #         outputs=model(input,date)
            #         outputs=outputs.squeeze()
                    
                #     target=label[:,-1,:]
                #     out=outputs[:,-1,:]
                #     #l2_sim = compute_similarity(outputs, label_base, method='cosine')
                #     _,predicted = torch.max(out.data,dim=1)
                #     #l2_sim = compute_similarity(target, label_base, method='cosine')
                #     _,llabels=torch.max(target.data,dim=1)
                    
                #     shape=torch.ones(batch_size)
                #     shape=shape.view(-1,1)
                #     predicted=predicted.view_as(shape)
                #     llabels=llabels.view_as(shape)
                #     total += target.size(0)
                #     correct +=(predicted == llabels).sum().item()
                #     for x,y in zip(predicted,llabels):
                #         relectcorrect+=(int(x.item()<4 and y.item()<4)+int(x.item()>3 and y.item()>3))
                # temp=best
                # best=relectcorrect / total if((relectcorrect / total)>best) else best
                # print('Accuracy on test set: %d %%' % (100 * correct / total))
                # print('reAccuracy on test set: %d %%' % (100 * relectcorrect / total))
        print(best)
        idx+=1
