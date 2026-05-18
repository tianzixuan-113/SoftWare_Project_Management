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

#change the LinerModule.py -B2 -initial

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
            a=self.rarelabels[batch][(number-self.size//4)//(self.size//8)]
            if(a==[]):
                num=random.randint(self.datalength,self.size//4)
            else:
                num=random.choice(self.rarelabels[batch][(number-self.size//4)//(self.size//8)])
            return torch.from_numpy(self.Mm1.fit_transform(self.data[batch,num-self.datalength:num,])),torch.from_numpy(self.Mm0.fit_transform(self.data[batch][num-self.datalength:num])),torch.from_numpy(self.label[batch][num])
        else:
            return torch.from_numpy(self.Mm1.fit_transform(self.data[batch][number:number+self.datalength])),torch.from_numpy(self.Mm0.fit_transform(self.data[batch][number:number+self.datalength])),torch.from_numpy(self.label[batch][number+self.datalength+1])
    
    def __len__(self):
        return self.len
 
class LinerNN(nn.Module):
    def __init__(self,length):
        super(LinerNN,self).__init__()
        self.l=length
        self.l0=nn.Linear(100*length,1024)
        # self.l1=nn.Linear(5169,2048)
        # self.l2=nn.Linear(2048,1024)
        self.l3=nn.Linear(1024,512)
        self.l4=nn.Linear(512,256)
        self.l5=nn.Linear(256,128)
        self.l6=nn.Linear(128,64)
        self.l7=nn.Linear(64,8)
        self.l0_=nn.Linear(100*length,1024)
        # self.l1_=nn.Linear(5169,2048)
        # self.l2_=nn.Linear(2048,1024)
        self.l3_=nn.Linear(1024,512)
        self.l4_=nn.Linear(512,256)
        self.l5_=nn.Linear(256,128)
        self.l6_=nn.Linear(128,64)
        self.l7_=nn.Linear(64,8)
        
        self.lout=nn.Linear(128,64)
        self.lout1=nn.Linear(64,8)
        
    def forward(self,x,y):
        x=x.view(10,-1)
        #y=y.view(10,-1)
        x=f.leaky_relu(self.l0(x))
        #y=f.leaky_relu(self.l0_(y))
        # x=f.leaky_relu(self.l1(x))
        # x=f.leaky_relu(self.l2(x))
        x=f.leaky_relu(self.l3(x))
        x=f.leaky_relu(self.l4(x))
        x=f.leaky_relu(self.l5(x))
        x=self.l6(x)
        # y=f.leaky_relu(self.l3_(y))
        # y=f.leaky_relu(self.l4_(y))
        # y=f.leaky_relu(self.l5_(y))
        # y=self.l6_(y)
        out=f.leaky_relu(x)
        # out=f.leaky_relu(self.lout(out))
        out=self.lout1(out)
        # y=self.l7(y)
        return out
        
train_dataset=StockDataset(['F:/stockPre/data/better_day_data/_0_better'],20000,False,21,False)
train_loader=DataLoader(train_dataset,shuffle=True,batch_size=10)
test_dataset=StockDataset(['F:/stockPre/data/better_day_data_test/_0_better'],3200,False,21,True)
test_loader=DataLoader(test_dataset,shuffle=True,batch_size=10)    
        
model=LinerNN(21)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
 


        
if __name__ =='__main__':
    #最优性能：归一化   57%    标准化   64%   
    #betterCsv('F:/stockPre/data/day_data/train.csv','F:/stockPre/data/day_data/train_liner.csv')
    #BinLabelConstruct('F:/stockPre/data/day_data/train_liner.csv','F:/stockPre/data/day_data/train_label_liner_bin.csv',True)
    #betterCsv('F:/stockPre/data/day_data/text.csv','F:/stockPre/data/day_data/text_liner.csv')
    #BinLabelConstruct('F:/stockPre/data/day_data/text_liner.csv','F:/stockPre/data/day_data/text_label_liner_bin.csv',False)
    torch.manual_seed(0)
    
    
   
    # l=[0.024759284731774415,0.050108076242876796,0.015720180782078996,0.42857142857142855,0.3953625466692867,0.06661426606405974, 0.014541167223423069,0.004323049715071723]
    # l=[0.42857142857142855/x for x in l]
    # sum=0
    # for x in l:
    #     sum+=x
    # #l=[x/sum for x in l]
    # l=[2.5,2.5,2.5,10,10,2.5,2.5,10]
    # l=np.array(l)
    # l=torch.from_numpy(l)

    best=0
    for epoch in range(20):
        running_loss = 0.0
        correct0 = 0
        relectcorrect0=0
        total0 = 0
        for batch_idx, data in enumerate(train_loader, 0):
            
            inputs0,inputs1, labels = data
            optimizer.zero_grad()
            # forward + backward + update
            outputs = model(inputs0,inputs1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 400 == 399:
                print('[%d,%5d] loss: %.3f'% (epoch + 1, batch_idx + 1, running_loss / (batch_idx+1)))
                running_loss = 0.0
                
            _,predicted = torch.max(outputs.data,dim=1)
            _,llabels=torch.max(labels.data,dim=1)
            #shape=torch.tensor([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])
            predicted=predicted.view(10,-1)
            llabels=llabels.view(10,-1)
            total0 += labels.size(0)
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
                images0,images1, labels = data
                outputs = model(images0,images1)
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
