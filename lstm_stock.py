from time import time
from datetime import datetime
import numpy as np
import pandas as pd
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
# Sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pickle

# Neural Networks
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger

# Plotting
import matplotlib.pyplot as plt
import json
import os
from os import path
import yfinance as yf
import math

class TimeseriesDataset(Dataset):
    '''
    Custom Dataset subclass.
    Serves as input to DataLoader to transform X
      into sequence data using rolling window.
    DataLoader using this dataset will output batches
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs.
    '''
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        return (self.X[index:index+self.seq_len], self.y[index+self.seq_len-1])


class MyDataModule(pl.LightningDataModule):
    '''
    PyTorch Lighting DataModule subclass:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

    Serves the purpose of aggregating all data loading
      and processing work in one place.
    '''

    def __init__(self, sym01, sym02,period, seq_len=1, batch_size=128, num_workers=0):
        super().__init__()
        self.reset(sym01, sym02, period, seq_len=1, batch_size=128, num_workers=0)

    def reset(self, sym01, sym02, period, seq_len=1, batch_size=128, num_workers=0):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.columns = None
        self.preprocessing = None
        self.testsize = 0
        self.sym01 = sym01
        self.sym02 = sym02
        self.period = period
        self.isreset = 1
        self.lastdata = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        '''
        Data is resampled to hourly intervals.
        Both 'np.nan' and '?' are converted to 'np.nan'
        'Date' and 'Time' columns are merged into 'dt' index
        '''
        print("reset is:",self.isreset)
        if stage == 'fit' and self.X_train is not None:
            print("pass setup", self.isreset)
            return
        if stage == 'test' and self.X_test is not None:
            print("pass setup", self.isreset)
            return
        #if stage is None and self.X_train is not None and self.X_test is not None:
            #return
        custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
        dir = '/Users/kahingleung/PycharmProjects/mylightning/'
        if path.exists(dir+self.sym01+'.csv'):
            print("reading file",dir+self.sym01+'.csv')
            hist1 = pd.read_csv(dir+self.sym01+'.csv',header=0,parse_dates=['Date'],date_parser=custom_date_parser)
            #hist1 = pd.read_csv(dir+sym01+'.csv',header=0)
            #print(hist1)
        else:
            ticker1 = yf.Ticker(self.sym01)
            hist1 = ticker1.history(period=self.period)
            hist1 = hist1.dropna().reset_index()
            hist1.to_csv(dir+self.sym01+'.csv',index=False)
        if path.exists(dir+self.sym02+'.csv'):
            print("reading file",dir+self.sym02+'.csv')
            hist2 = pd.read_csv(dir+self.sym02+'.csv',header=0,parse_dates=['Date'],date_parser=custom_date_parser)
            #hist2 = pd.read_csv(dir+sym02+'.csv',header=0)
            #print(hist2)
        else:
            ticker2 = yf.Ticker(self.sym02)
            hist2 = ticker2.history(period=self.period)
            hist2 = hist2.dropna().reset_index()
            hist2.to_csv(dir+self.sym02 + '.csv', index=False)
        df = hist1.merge(hist2, left_on='Date', right_on='Date').reset_index()
        #df['date'] = df['Date'].apply(lambda x: datetime.strptime(x, "%d/%m/%Y"))
        #df['year-month'] = df['Date'].apply(lambda x: int((x.year - y) * 12 + x.month))
        df['year'] = df['Date'].apply(lambda x: int(x.year))
        df['month'] = df['Date'].apply(lambda x: int(x.month))
        df['day'] = df['Date'].apply(lambda x: int(x.day))
        df['log-vol'] = df['Volume_x'].apply(lambda x: math.log(1 + x))
        tgt = 'close-y-next-diff'
        df[tgt] = df['Close_y'].shift(-1)
        df[tgt] = df[[tgt, 'Close_y']].apply(lambda x: (x[tgt] - x['Close_y'])*100/x['Close_y'], axis=1)
        print(df.columns)
        lag=5
        for f in ['Open_x','Close_x','High_x','Low_x','log-vol']:
            for i in range(1,lag+1):
                col = f +'-over-lag-'+ str(i)
                df[col] = df[f].shift(i)
                df[col] = df[[col,f]].apply(lambda x: (x[f]-x[col])*100/x[col], axis=1)
        print("last date is",df['Date'].iloc[-1])
        last = df.iloc[-1*self.seq_len:].copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index()
        pca_features = [f for f in df.columns if 'lag' in f]
        print("pca features:",pca_features)
        X = df[pca_features].values
        pca = PCA(n_components=3)
        pca.fit(X)
        X_reduced = pca.transform(X)
        with open(self.sym01+'_pca.pkl', 'wb') as pickle_file:
            pickle.dump(pca, pickle_file)
        #print("pca transform",last[pca_features].values.reshape(1,-1))
        last_pca = pca.transform(last[pca_features].values)
        self.lastdata = last_pca
        print("pca explained var",sum(pca.explained_variance_ratio_))
        print("last pca", last_pca)
        pf = pd.DataFrame(X_reduced, columns=['PCA1', 'PCA2','PCA3'])
        print("pca len",len(pf))
        df['PCA1'] = pf['PCA1']
        df['PCA2'] = pf['PCA2']
        df['PCA3'] = pf['PCA3']
        print("df len",len(df.index))
        #print("last data is",df[['Date','PCA1','PCA2','PCA3',tgt]].iloc[-5:])
        n = len(df.index)
        t = int(n*0.8)
        #features = ['PCA1', 'PCA2', 'PCA11', 'PCA22', 'PCA12', 'log-vol']
        features = ['PCA1', 'PCA2', 'PCA3']
        #features = ['Close_x', 'Open_x', 'High_x', 'Low_x', 'Close_y', 'log-vol']
        label = [tgt]

        xtrain = df[features + label].iloc[:t]
        xtest = df[features + label].iloc[t:]
        #print("data size",xtrain.size, xtest.size)

        X_train = xtrain.iloc[:-50]
        X_val = xtrain.iloc[-50:]

        if stage == 'fit' or stage is None:
            self.X_train = X_train[features].values
            self.y_train = X_train[label].values.reshape((-1, 1))
            self.X_val = X_val[features].values
            self.y_val = X_val[label].values.reshape((-1, 1))

        if stage == 'test' or stage is None:
            self.X_test = xtest[features].values
            self.y_test = xtest[label].values.reshape((-1, 1))

    def train_dataloader(self):
        train_dataset = TimeseriesDataset(self.X_train,
                                          self.y_train,
                                          seq_len=self.seq_len)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers)

        return train_loader

    def val_dataloader(self):
        val_dataset = TimeseriesDataset(self.X_val,
                                        self.y_val,
                                        seq_len=self.seq_len)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers)

        return val_loader

    def test_dataloader(self):
        #print(self.X_test)
        #print(self.y_test)
        test_dataset = TimeseriesDataset(self.X_test,
                                         self.y_test,
                                         seq_len=self.seq_len)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers)

        return test_loader

    def today_dataloader(self):
        #print(self.X_test)
        #print(self.y_test)
        today_dataset = TimeseriesDataset(self.lastdata,
                                         np.zeros(self.seq_len),
                                         seq_len=self.seq_len)
        today_loader = DataLoader(today_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=self.num_workers)

        return today_loader


class LSTMRegressor(pl.LightningModule):
    '''
    Standard PyTorch Lightning module:
    https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    '''

    def __init__(self,
                 n_features,
                 hidden_size,
                 seq_len,
                 batch_size,
                 num_layers,
                 dropout,
                 learning_rate,
                 criterion):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        #y_pred = torch.relu(y_pred)
        return y_pred

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        #print("test_step gets",x,y)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        return loss
    def validation_epoch_end(self, outputs):
        avg_loss = sum(outputs)/len(outputs)
        self.log("ptl/val_loss", avg_loss)
    def test_epoch_end(self, outputs):
        avg_loss = sum(outputs)/len(outputs)
        self.log("ptl/test_loss", avg_loss)
'''
All parameters are aggregated in one place.
This is useful for reporting experiment params to experiment tracking software
'''
def myTrain(config,num_epochs,sym01,sym02,period):
    p = dict(
        seq_len = config['seq_len'],
        batch_size = config['batch_size'],
        criterion = nn.MSELoss(),
        max_epochs = num_epochs,
        n_features = 3,
        hidden_size = config['hidden_size'],
        num_layers = config['num_layers'],
        dropout = config['dropout'],
        learning_rate = config['lr']
    )
    print("myTrain parameters:",sym01,sym02,period)

    seed_everything(1)

    csv_logger = CSVLogger('./', name='lstm', version='0'),
    metrics = {"loss": "ptl/val_loss"}
    trainer = Trainer(
        max_epochs=p['max_epochs'],
        logger=csv_logger,
        callbacks=[TuneReportCallback(metrics, on="validation_end")]
        #gpus=1,
        #row_log_interval=1,
        #progress_bar_refresh_rate=2,
    )
    model = LSTMRegressor(
        n_features = p['n_features'],
        hidden_size = p['hidden_size'],
        seq_len = p['seq_len'],
        batch_size = p['batch_size'],
        criterion = p['criterion'],
        num_layers = p['num_layers'],
        dropout = p['dropout'],
        learning_rate = p['learning_rate']
    )

    dm = MyDataModule(
        sym01=sym01,
        sym02=sym02,
        period=period,
        seq_len = p['seq_len'],
        batch_size = p['batch_size']
    )
    dm.reset(
        sym01=sym01,
        sym02=sym02,
        period=period,
        seq_len = p['seq_len'],
        batch_size = p['batch_size']
    )
    dm.setup('test')
    trainer.fit(model, dm)
    testresult = trainer.test(model, datamodule=dm)
    trainer.save_checkpoint(sym01+"-lstm.ckpt")
    print(testresult)
    return model



def hypertune(num_samples, num_epochs, sym01, sym02, period):
    config = {
        "seq_len": tune.choice([5, 10]),
        "hidden_size": tune.choice([10, 50, 100]),
        "batch_size": tune.choice([30,60]),
        "dropout": tune.choice([0.1, 0.2]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "num_layers": tune.choice([2, 3, 4])
    }
    trainable = tune.with_parameters(
        myTrain,
        num_epochs=num_epochs,
        sym01=sym01,
        sym02=sym02,
        period=period,
    )
    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": 1,
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        name="tune_lstm")
    print("tuning finished")
    return analysis.best_config

def getAcc(p,y):
    n = len(p)
    total = 0
    hit = 0
    for i in range(n):
        total = total + 1
        if p[i][0]*y[i][0] > 0:
            hit = hit + 1

    return hit*100/total
def buyAcc(p,y):
    n = len(p)
    total = 0
    hit = 0
    bal = 0
    for i in range(n):
        if p[i][0] > 0:
            total = total + 1
            if y[i][0] > 0:
                hit = hit + 1
                delta = min(y[i][0],3)
            else:
                delta = max(y[i][0],-3)

            bal = bal + delta

    if total == 0 :
        return 0
    return bal/total

def mainTest(num_samples,num_epochs,sym01,sym02,period):
    best_config = hypertune(num_samples,num_epochs,sym01,sym02,period)
    with open(sym01+'-best.json','w') as outfile:
        json.dump(best_config,outfile)
    print(best_config)
    model = myTrain(best_config,num_epochs*10,sym01,sym02,period)
    model.eval()
    model.freeze()
    print("testing final result")
    testdm = MyDataModule(
        sym01=sym01,
        sym02=sym02,
        period=period,
        seq_len=best_config['seq_len'],
        batch_size=best_config['batch_size']
    )
    testdm.reset(
        sym01=sym01,
        sym02=sym02,
        period=period,
        seq_len = best_config['seq_len'],
        batch_size = best_config['batch_size']
    )
    testdm.setup('test')
    loader = testdm.test_dataloader()
    mae = []
    mse = []
    predictions =[]
    truth = []
    for xtest, labels in iter(loader):
        y_hat = model(xtest)
        mae.append(mean_absolute_error(labels,y_hat))
        mse.append(mean_squared_error(labels,y_hat))
        predictions.append(y_hat)
        truth.append(labels)

    p=np.concatenate(predictions)
    y=np.concatenate(truth)
    print("final result")
    print("mae =",mean_absolute_error(y,p))
    print("mse =",mean_squared_error(y,p))
    score = buyAcc(p,y)
    acc = getAcc(p,y)
    print("acc =",score)
    plt.clf()
    plt.plot(p,'r')
    plt.plot(y,'y')
    #plt.show()
    plt.savefig(sym01+'-'+sym02+'-test.png')

    loader = testdm.today_dataloader()
    for today, labels in iter(loader):
        print("predict tmr from ",today)
        y_tmr = model(today)
        y_tmr = y_tmr[0][0].numpy()
        print("tmr return is", y_tmr)
    return (score,acc,y_tmr)


#
#
#
# Main
#
#
#
#sym01 = '2388.HK'
mkt='HK'
period = '3y'
num_samples = 10
num_epochs = 10
dependency = []
if mkt == 'HK':
    sym02 = '7200.HK'
    #sym02 = '7500.HK'
    #sym02 = '^HSI'
    list=[5,2318,1398,2628,823,700,1810,175,3690,2269]
    for i in list:
        dependency.append("{:04d}.HK".format(i))
else:
    sym02 = 'NASDX'
    #sym02 = 'QLD'
    #sym02 = 'QID'
    dependency=['FB', 'AAPL', 'AMZN', 'GOOG', 'NFLX', 'SQ', 'MTCH', 'AYX', 'ROKU', 'TTD' ]
outcome = []
df = pd.DataFrame()
print(dependency)
for sym01 in dependency:
    score,acc,tmr = mainTest(num_samples,num_epochs,sym01,sym02,period)
    outcome.append({'symbol': sym01, 'score': score, 'acc':acc, 'tmr': tmr})

sorted_list = pd.DataFrame(outcome).sort_values('score')
print(sorted_list)
good = sorted_list[sorted_list.acc > 50]
if len(good) > 0 :
    print(sum(good[['score','tmr']].apply(lambda x: x['score']*x['tmr'],axis=1))/sum(good['score']))
