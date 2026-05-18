import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import csv

#change LinerModule_pre.py -B3 -initial

class StockDataset(Dataset):
    def __init__(self,data_path,size,isbin):
        self.data=[]
        self.label=[]
        self.size=size
        self.len=size*len(data_path)
        for path in data_path:
            df = pd.read_csv(f"{path}.csv",encoding="utf-8")
            df = np.array(df[['open','high','low','close','preclose','volume','amount','turn','pctChg','peTTM','pbMRQ','psTTM','pcfNcfTTM','tradestatus']],dtype='float32')#将pandas读取的数据转化为array
            x_data=torch.from_numpy(df)
            self.data.append(x_data)
            label_str=f'{path}_label_bin.csv' if(isbin) else f'{path}_label.csv'
            df = pd.read_csv(label_str,encoding="utf-8")
            df = np.array(df,dtype='float32')#将pandas读取的数据转化为array
            y_data=torch.from_numpy(df)
            self.label.append(y_data)
        
    def __getitem__(self, index):
        return self.data[index//self.size][index%self.size:index%self.size+30],self.data[index//self.size][index%self.size+31][:4]
    
    def __len__(self):
        return self.len
 
class LinerNN(nn.Module):
    def __init__(self):
        super(LinerNN,self).__init__()
        self.l1=nn.Linear(420,256)
        self.l2=nn.Linear(256,128)
        self.l3=nn.Linear(128,64)
        self.l4=nn.Linear(64,4)
        self.p1=nn.PReLU(256)
        self.p2=nn.PReLU(128)
        
    def forward(self,x):
        x=x.view(-1,420)
        x=f.elu(self.l1(x))
        x=f.elu(self.l2(x))
        x=f.elu(self.l3(x))
        x=self.l4(x)
        return x
        
    
        
    


        
if __name__ =='__main__':
    #最优性能：39%
    #betterCsv('F:/stockPre/data/day_data/train.csv','F:/stockPre/data/day_data/train_liner.csv')
    #BinLabelConstruct('F:/stockPre/data/day_data/train_liner.csv','F:/stockPre/data/day_data/train_label_liner_bin.csv',True)
    #betterCsv('F:/stockPre/data/day_data/text.csv','F:/stockPre/data/day_data/text_liner.csv')
    #BinLabelConstruct('F:/stockPre/data/day_data/text_liner.csv','F:/stockPre/data/day_data/text_label_liner_bin.csv',False)
    torch.manual_seed(0)
    
    train_dataset=StockDataset(['F:/stockPre/data/day_data/train_better'],5000,False)
    train_loader=DataLoader(train_dataset,shuffle=True,batch_size=10)
    test_dataset=StockDataset(['F:/stockPre/data/day_data/text_better'],900,False)
    test_loader=DataLoader(test_dataset,shuffle=True,batch_size=10)
    
    model=LinerNN()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0009)
    best=0
    for epoch in range(20):
        running_loss = 0.0
        correct = 0
        relectcorrect=0
        total = 0
        for batch_idx, data in enumerate(train_loader, 0):
            
            inputs, labels = data
            optimizer.zero_grad()
            # forward + backward + update
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 100 == 99:
                print('[%d,%5d] loss: %.3f'% (epoch + 1, batch_idx + 1, running_loss / batch_idx))
                running_loss = 0.0
            
        correct=0        
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = model(images)
                total+=labels.size(0)
                for row,l in zip(outputs,labels):
                    if(row[2]==row[0] and row[3]==row[1] and row[0]<=row[3]):
                        z=torch.Tensor([0])    #low=start<end=high:1
                    elif(row[2]==row[0] and row[3]<=row[1] and row[0]<=row[3]):
                        z=torch.Tensor([1])    #low=start<end<high:2
                    elif(row[2]<=row[0] and row[3]==row[1] and row[0]<=row[3]):
                        z=torch.Tensor([2])     #low<start<end=high:3
                    elif(row[2]<=row[0] and row[3]<=row[1] and row[0]<=row[3]):
                        z=torch.Tensor([3])      #low<start<end<high:4
                    elif(row[2]==row[3] and row[0]==row[1] and row[0]>=row[3]):
                        z=torch.Tensor([4])      #low=end<start=high:5
                    elif(row[2]==row[3] and row[0]<=row[1] and row[0]>=row[3]):
                        z=torch.Tensor([5])      #low=end<start<high:6
                    elif(row[2]<=row[3] and row[0]==row[1] and row[0]>=row[3]):
                        z=torch.Tensor([6])      #low<end<start=high:7
                    elif(row[2]<=row[3] and row[0]<=row[1] and row[0]>=row[3]):
                        z=torch.Tensor([7])      #low<end<start<high:8
                    
                    if(l[2]==l[0] and l[3]==l[1] and l[0]<=l[3]):
                        z1=torch.Tensor([0])    #low=start<end=high:1
                    elif(l[2]==l[0] and l[3]<=l[1] and l[0]<=l[3]):
                        z1=torch.Tensor([1])    #low=start<end<high:2
                    elif(l[2]<=l[0] and l[3]==l[1] and l[0]<=l[3]):
                        z1=torch.Tensor([2])     #low<start<end=high:3
                    elif(l[2]<=l[0] and l[3]<=l[1] and l[0]<=l[3]):
                        z1=torch.Tensor([3])      #low<start<end<high:4
                    elif(l[2]==l[3] and l[0]==l[1] and l[0]>=l[3]):
                        z1=torch.Tensor([4])      #low=end<start=high:5
                    elif(l[2]==l[3] and l[0]<=l[1] and l[0]>=l[3]):
                        z1=torch.Tensor([5])      #low=end<start<high:6
                    elif(l[2]<=l[3] and l[0]==l[1] and l[0]>=l[3]):
                        z1=torch.Tensor([6])      #low<end<start=high:7
                    elif(l[2]<=l[3] and l[0]<=l[1] and l[0]>=l[3]):
                        z1=torch.Tensor([7])      #low<end<start<high:8
                    
                    correct +=(z == z1).sum().item()
                
                
            print(f'Accuracy on test set: {correct/total} ')
    print(best)
