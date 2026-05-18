import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import csv
import random
from sklearn.preprocessing import StandardScaler

# class StockDataset(Dataset):
#     def __init__(self,data_path,size,isbin):
#         self.data=[]
#         self.label=[]
#         self.size=size
#         self.len=size*len(data_path)
#         for path in data_path:
#             df = pd.read_csv(f"{path}.csv",encoding="utf-8")
#             df = np.array(df,dtype='float32')#将pandas读取的数据转化为array
#             x_data=torch.from_numpy(df)
#             self.data.append(x_data)
#             label_str=f'{path}_label_bin.csv' if(isbin) else f'{path}_label.csv'
#             df = pd.read_csv(label_str,encoding="utf-8")
#             df = np.array(df,dtype='float32')#将pandas读取的数据转化为array
#             y_data=torch.from_numpy(df)
#             self.label.append(y_data)
#         #self.data=torch.stack(self.data,dim=0)
        
#     def __getitem__(self, index):
#         return self.data[index//self.size][index%self.size:index%self.size+21],self.label[index//self.size][index%self.size+22]
    
#     def __len__(self):
#         return self.len

class StockDataset(Dataset):
    def __init__(self,data_path,size,isbin,datalength,istext=False):
        self.data=[]
        self.label=[]
        self.size=size
        self.len=size*len(data_path)
        self.rarelabels=[]
        self.datalength=datalength
        for path in data_path:
            self.Mm = StandardScaler()
            df = pd.read_csv(f"{path}.csv",encoding="utf-8")
            df = np.array(df,dtype='float32')#将pandas读取的数据转化为array
            # x_data=torch.from_numpy(df)
            self.data.append(df)
            label_str=f'{path}_label_bin.csv' if(isbin) else f'{path}_label.csv'
            df = pd.read_csv(label_str,encoding="utf-8")
            df = np.array(df,dtype='float32')#将pandas读取的数据转化为array
            # y_data=torch.from_numpy(df)
            self.label.append(df)
            rarelabel=[[],[],[],[],[],[]]
            with open(f'{path}_label.csv',"r+",encoding='utf-8',newline='') as d:
                    csv_reader = csv.reader(d)
                    for idx,row in enumerate(csv_reader,-1):
                        if(row[0]=='label0' or row[0]=='label' or idx<self.datalength or idx>(self.size//4)):
                            continue
                        if((row[0]=='0' and istext) or ((not istext) and row[0]=='1')):
                            rarelabel[0].append(idx)
                        elif((row[0]=='1' and istext) or ((not istext) and row[1]=='1')):
                            rarelabel[1].append(idx)
                        elif((row[0]=='2' and istext) or ((not istext) and row[2]=='1')):
                            rarelabel[2].append(idx)
                        elif((row[0]=='5' and istext) or ((not istext) and row[5]=='1')):
                            rarelabel[3].append(idx)
                        elif((row[0]=='6' and istext) or ((not istext) and row[6]=='1')):
                            rarelabel[4].append(idx)
                        elif((row[0]=='7' and istext) or ((not istext) and row[7]=='1')):
                            rarelabel[5].append(idx)
            self.rarelabels.append(rarelabel)
        
    def __getitem__(self, index):
        batch=index//self.size#得到批次数
        number=index%self.size#得到该批次的索引
        if(number>(self.size//4)):
            num=random.choice(self.rarelabels[batch][(number-self.size//4)//(self.size//8)])
            return torch.from_numpy(self.Mm.fit_transform(self.data[batch][num-self.datalength:num])),torch.from_numpy(self.label[batch][num])
        else:
            return torch.from_numpy(self.Mm.fit_transform(self.data[batch][number:number+self.datalength])),torch.from_numpy(self.label[batch][number+self.datalength+1])
    
    def __len__(self):
        return self.len
 
class rnnModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers):
        super(rnnModel, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn =torch.nn.RNN(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=num_layers,nonlinearity='relu',batch_first=True,bidirectional=True)
        # self.l0=[[nn.Linear(64,32),nn.Linear(32,8)],
        #          [nn.Linear(64,32),nn.Linear(32,8)],
        #          [nn.Linear(64,32),nn.Linear(32,8)],
        #          [nn.Linear(64,32),nn.Linear(32,8)]]
        # self.l1=nn.Linear(32,16)
        # self.l2=nn.Linear(16,8)
        self.l0=nn.Linear(2121,1024)
        self.l1=nn.Linear(1024,512)
        self.l2=nn.Linear(512,256)
        self.l3=nn.Linear(256,128)
        self.l4=nn.Linear(128,64)
        self.l5=nn.Linear(64,8)
    def forward(self, input):
        hidden = torch.zeros(self.num_layers*2,self.batch_size,self.hidden_size)
        out,hidden = self.rnn(input, hidden)
        out=out.contiguous().view(21,10,2,101)
        out0=out[:,:,0]
        out1=out[:,:,1]
        out=out0+out1
        # output=[out[x*2]+out[x*2+1] for x in range(4)]
        # out=[]
        # for idx,o in enumerate(output,0):
        #     o=f.relu(self.l0[idx][0](o))
        #     o=self.l0[idx][1](o)
        #     out.append(o)
        # out=torch.stack(out,dim=0)
        # out=out.view(self.batch_size,-1)
        # out=f.relu(self.l1(out))
        # out=self.l2(out)
        x=out.contiguous().transpose(0,1)
        x=x.contiguous().view(self.batch_size,-1)
        x=f.leaky_relu(self.l0(x))
        x=f.leaky_relu(self.l1(x))
        x=f.leaky_relu(self.l2(x))
        x=f.leaky_relu(self.l3(x))
        x=f.leaky_relu(self.l4(x))
        x=self.l5(x)
        return x

train_dataset=StockDataset(['F:/stockPre/data/ddd/_0_better'],20000,False,21,False)
train_loader=DataLoader(train_dataset,shuffle=True,batch_size=10)
test_dataset=StockDataset(['F:/stockPre/data/ddd/_1_better'],3600,False,21,True)
test_loader=DataLoader(test_dataset,shuffle=True,batch_size=10)

model=rnnModel(101,101,10,1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)       
    
        
    


        
if __name__ =='__main__':
    #最优性能：39%
    # train_dataset=StockDataset(['F:/stockPre/data/better_day_data/_0_better',
    #                             'F:/stockPre/data/better_day_data/_1_better',
    #                             'F:/stockPre/data/better_day_data/_2_better',
    #                             'F:/stockPre/data/better_day_data/_3_better',
    #                             'F:/stockPre/data/better_day_data/_4_better',
    #                             'F:/stockPre/data/better_day_data/_5_better',
    #                             'F:/stockPre/data/better_day_data/_6_better',
    #                             'F:/stockPre/data/better_day_data/_7_better',
    #                             'F:/stockPre/data/better_day_data/_8_better',
    #                             'F:/stockPre/data/better_day_data/_9_better',],5000,False)

    
    
    
    for epoch in range(10):
        running_loss = 0.0
        correct0 = 0
        relectcorrect0=0
        total0 = 0
        for batch_idx, data in enumerate(train_loader, 0):           
            inputs, target = data
            optimizer.zero_grad()
            # forward + backward + update
            outputs = model(inputs)            
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 400 == 399:
                print('[%d,%5d] loss: %.3f'% (epoch + 1, batch_idx + 1, running_loss / batch_idx))
                running_loss = 0.0
            
            _,predicted = torch.max(outputs.data,dim=1)
            _,llabels=torch.max(target.data,dim=1)
            shape=torch.tensor([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])
            predicted=predicted.view_as(shape)
            llabels=llabels.view_as(shape)
            total0 += target.size(0)
            correct0 +=(predicted == llabels).sum().item()
            for x,y in zip(predicted,llabels):
                    relectcorrect0+=(int(x.item()<4 and y.item()<4)+int(x.item()>3 and y.item()>3))
        print('Accuracy on test set: %d %%' % (100 * correct0 / total0))
        print('reAccuracy on test set: %d %%' % (100 * relectcorrect0 / total0))
                
        correct = 0
        relectcorrect=0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, Labels = data
                outputs = model(images)
                _,Predicted = torch.max(outputs.data,dim=1)
                Predicted=Predicted.view_as(Labels)
                total += Labels.size(0)
                correct +=(Predicted == Labels).sum().item()
                for x,y in zip(predicted,Labels):
                    relectcorrect+=(int(x.item()<4 and y.item()<4)+int(x.item()>3 and y.item()>3))
            print('Accuracy on test set: %d %%' % (100 * correct / total))
            print('reAccuracy on test set: %d %%' % (100 * relectcorrect / total))
