import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch import nn
import numpy as np
from torch.utils.data import DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):

        # save which are the features and target data and sequence length
        self.features = features
        self.target = target
        self.sequence_length = sequence_length

        # load data from dataframe into tensor
        y = torch.tensor(dataframe[self.target].values).float()
        X = torch.tensor(dataframe[self.features].values).float()

        # remove padding from data and save data in class
        self.y = y[sequence_length:]
        self.X = X[sequence_length:,:]

        # make extra dataseries for padding
        self.yPadding = y[:sequence_length*2]
        self.XPadding = X[:sequence_length*2,:]
        self.sequence_length = sequence_length


    def __len__(self):

        # return length shape
        return self.X.shape[0]


    def __getitem__(self, i): 

        # select data from the dataset without padding and return data
        if i > self.sequence_length:
            Xr = self.X[i-self.sequence_length:i,:]
            yr = self.y[i-1]

            return Xr , yr
          
        # select from the padding dataset and return data
        else:
            Xr = self.XPadding[i:i+self.sequence_length,:]
            yr = self.yPadding[i+self.sequence_length - 1]

            return Xr, yr


class LSTM(nn.Module):
    def __init__(self, num_features, hidden_units):
        super().__init__()

        # save needed info to initialize layers
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.num_layers = 1

        # first LSTM layer of model
        self.lstm1 = nn.LSTM(input_size=num_features,
                             hidden_size=hidden_units,
                             batch_first=True,
                             num_layers=self.num_layers)

        # second LSTM layer of model
        self.lstm2 = nn.LSTM(input_size=hidden_units,
                             hidden_size=hidden_units,
                             batch_first=True,
                             num_layers=self.num_layers)

        # dropout layer
        self.dropout = nn.Dropout(p=0.1)

        # linear layer
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)


    def forward(self, x):
        batch_size = x.shape[0]

        # initialize c0 and h0
        c0 = torch.zeros(self.num_layers, batch_size, 
                         self.hidden_units).requires_grad_()
        h0 = torch.zeros(self.num_layers, batch_size, 
                         self.hidden_units).requires_grad_()
        
        # load data into layers as written in the paper and return the output 
        # data
        x, (_, _) = self.lstm1(x, (h0, c0))
        x = self.dropout(x) 
        _, (ht,_) = self.lstm2(x, (h0, c0)) 
        out = self.linear(ht[0]).flatten() 

        return out


class HydroLSTM():
    def __init__(self, df_train, df_test, target, features, sequence_length, batch_size, epochs,
               num_hidden_units, learning_rate):

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.target = target
        self.features = features
        self.epochs = epochs
        self.num_hidden_units = num_hidden_units
        self.learning_rate = learning_rate
        self.df_train = df_train
        self.df_test = df_test

        # make placeholder for model output data
        self.y_pred_col = 'LSTM model discharge prediction'

        # make train and test dataset
        self.train_dataset = TimeSeriesDataset(df_train,
                                               target=self.target,
                                               features=self.features,
                                               sequence_length=self.sequence_length)
        self.test_dataset = TimeSeriesDataset(df_test,
                                              target=self.target,
                                              features=self.features,
                                              sequence_length=self.sequence_length)

        # load dataset in DataLoader
        self.train_loader = DataLoader(self.train_dataset, 
                                  batch_size=self.batch_size, 
                                  shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, 
                                batch_size=self.batch_size, 
                                shuffle=False)
        self.train_eval_loader = DataLoader(self.train_dataset, 
                                      batch_size=self.batch_size,  
                                      shuffle=False)

        # initialize model 
        self.model = LSTM(num_features=len(self.features), hidden_units=self.num_hidden_units)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def __train_model(self):

        # initialize variables
        num_batches = len(self.train_loader)
        total_loss = 0
        
        # iterate through data from dataloader and run model and calculate loss
        self.model.train()
        for X, y in self.train_loader:
            output = self.model(X)
            loss = self.loss_function(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / num_batches
        print(f'Train loss: {avg_train_loss}')
        return avg_train_loss
      

    def __test_model(self):
        
        # initialize variables
        num_batches = len(self.test_loader)
        total_loss = 0

        # iterate through data from dataloader and test model and calculate loss
        self.model.eval()
        with torch.no_grad():
            for X, y in self.test_loader:
                output = self.model(X)
                total_loss += self.loss_function(output, y).item()

        avg_test_loss = total_loss / num_batches
        print(f'Test loss: {avg_test_loss}')
        return avg_test_loss

    def __predict(self, data_loader):
        
        # initiaze output data
        output = torch.tensor([])

        # iterate through data from data loader and make prediction
        self.model.eval()
        with torch.no_grad():
            for X, _ in data_loader:
                y_pred = self.model(X)
                output = torch.cat((output, y_pred), 0)
        output = output
        return output

    def train(self):

        # run untrained test and make loss lists
        print('Untrained test\n--------')
        self.train_losses = []
        self.test_losses = [self.__test_model()]
        print()

        # # train model and calculate training and test loss
        for epoch_i in range(1,self.epochs+1):
            print(f'Epoch {epoch_i}\n-----')

            # epoch
            train_loss = self.__train_model()
            test_loss = self.__test_model()
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

    def predict_using_train_data(self):
        # predict data with model
        y_pred_train = self.__predict(self.train_eval_loader).numpy()

        # load data in dataframe
        df_train_out = pd.DataFrame(index=self.df_train.index[sequence_length:],
                                data={target:self.df_train[target][sequence_length:],
                                      self.y_pred_col:y_pred_train})
        
        return df_train_out
  
    def predict_using_test_data(self):
        # predict data with model
        y_pred_test = self.__predict(self.test_loader).numpy()

        # load data in dataframe
        df_test_out = pd.DataFrame(index=self.df_test.index[sequence_length:],
                                data={target:self.df_test[target][sequence_length:],
                                      self.y_pred_col:y_pred_test})

        return df_test_out

    