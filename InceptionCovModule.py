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
        
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_1 = nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16,24,kernel_size=5, padding=2)
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16,24,kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24,24,kernel_size=3, padding=1)
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        branch_pool = f.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels,kernel_size=3, padding=1)
    def forward(self, x):
        y = f.leaky_relu(self.conv1(x))
        y = self.conv2(y)
        return f.leaky_relu(x + y)
    
class CovNN(nn.Module):
    def __init__(self):
        super(CovNN,self).__init__()
        self.cov1=nn.Conv2d(1,10,kernel_size=4)
        self.cov2=nn.Conv2d(88,20,kernel_size=2)
        self.r1=ResidualBlock(10)
        self.r2=ResidualBlock(20)
        self.incep1=InceptionA(10)
        self.incep2=InceptionA(20)
        self.pooling=nn.MaxPool2d(2)
        self.l1=nn.Linear(8448,2048)
        self.l2=nn.Linear(2048,512)
        self.l3=nn.Linear(512,128)
        self.l4=nn.Linear(128,32)
        self.l5=nn.Linear(32,8)
        
    def forward(self,x):
        batch_size=x.size(0)
        x=f.leaky_relu(self.pooling(self.cov1(x)))
        x=self.r1(x)
        x=self.incep1(x)
        x=f.leaky_relu(self.pooling(self.cov2(x)))
        x=self.r2(x)
        x=self.incep2(x)
        x=x.view(batch_size,-1)
        x=f.leaky_relu(self.l1(x))
        x=f.leaky_relu(self.l2(x))
        x=f.leaky_relu(self.l3(x))
        x=f.leaky_relu(self.l4(x))
        x=self.l5(x)
        return x
        
train_dataset=StockDataset(['F:/stockPre/data/ddd/_0_better'],20000,False,21,False)
train_loader=DataLoader(train_dataset,shuffle=True,batch_size=10)
test_dataset=StockDataset(['F:/stockPre/data/ddd/_1_better'],3200,False,21,True)
test_loader=DataLoader(test_dataset,shuffle=True,batch_size=10)
    
model=CovNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)          
        
    


        
if __name__ =='__main__':
    #最优性能：61%
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
                print('[%d,%5d] loss: %.3f'% (epoch + 1, batch_idx + 1, running_loss / 1000))
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
