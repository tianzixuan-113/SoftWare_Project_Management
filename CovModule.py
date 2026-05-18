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
from sklearn.preprocessing import MinMaxScaler

class StockDataset(Dataset):
    def __init__(self,data_path,size,isbin,datalength,istext=False):
        self.data=[]
        self.label=[]
        self.size=size
        self.len=size*len(data_path)
        self.rarelabels=[]
        self.datalength=datalength
        self.Mm0 = MinMaxScaler()
        self.Mm1 = StandardScaler()
        for path in data_path:
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
            return torch.from_numpy(self.Mm1.fit_transform(self.data[batch][num-self.datalength:num])).unsqueeze(0),torch.from_numpy(self.Mm0.fit_transform(self.data[batch][num-self.datalength:num])),torch.from_numpy(self.label[batch][num])
        else:
            return torch.from_numpy(self.Mm1.fit_transform(self.data[batch][number:number+self.datalength])).unsqueeze(0),torch.from_numpy(self.Mm0.fit_transform(self.data[batch][number:number+self.datalength])),torch.from_numpy(self.label[batch][number+self.datalength+1])
    
    def __len__(self):
        return self.len
 
class CovNN(nn.Module):
    def __init__(self):
        super(CovNN,self).__init__()
        self.cov1=nn.Conv2d(1,10,kernel_size=4)
        self.cov2=nn.Conv2d(10,20,kernel_size=2)
        self.pooling=nn.MaxPool2d(2)
        self.l0=nn.Linear(1920,512)
        self.l1=nn.Linear(512,128)
        self.l2=nn.Linear(128,32)
        self.l3=nn.Linear(32,8)
        
    def forward(self,x):
        batch_size=x.size(0)
        x=f.tanh(self.pooling(self.cov1(x)))
        x=f.tanh(self.pooling(self.cov2(x)))
        x=x.view(batch_size,-1)
        x=f.leaky_relu(self.l0(x))
        x=f.leaky_relu(self.l1(x))
        x=f.leaky_relu(self.l2(x))
        x=self.l3(x)
        
        return x
        

train_dataset=StockDataset(['F:/stockPre/data/ddd/_0_better'],20000,False,21,False)
train_loader=DataLoader(train_dataset,shuffle=True,batch_size=10)
test_dataset=StockDataset(['F:/stockPre/data/ddd/_1_better'],3200,False,21,True)
test_loader=DataLoader(test_dataset,shuffle=True,batch_size=10)
    
model=CovNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)        
    


        
if __name__ =='__main__':
    #最优性能：58%
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
    best=0
    for epoch in range(20):
        running_loss = 0.0
        correct0 = 0
        relectcorrect0=0
        total0 = 0
        for batch_idx, data in enumerate(train_loader, 0):           
            inputs,_, target = data
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
                images,_, labels = data
                outputs = model(images)
                _,predicted = torch.max(outputs.data,dim=1)
                predicted=predicted.view_as(labels)
                total += labels.size(0)
                correct +=(predicted == labels).sum().item()
                for x,y in zip(predicted,labels):
                    relectcorrect+=(int(x.item()<4 and y.item()<4)+int(x.item()>3 and y.item()>3))
            best=relectcorrect / total if((relectcorrect / total)>best) else best
            print('Accuracy on test set: %d %%' % (100 * correct / total))
            print('reAccuracy on test set: %d %%' % (100 * relectcorrect / total))
    print(best)
