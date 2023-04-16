import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

class Model(nn.Module):

    def __init__(self, loss_fn=nn.BCELoss(), optimizer=optim.Adam, lr=0.00001, n_epochs=100, batch_size=10):
        self.hidden1 = nn.Linear(85, 128)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(128, 50)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(50, 24)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(24, 1)
        self.act_output = nn.Sigmoid()

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
    
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act_output(self.output(x))
        return x

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        loss_fn = self.loss_fn
        optimizer = self.optimizer(self.parameters(), lr=self.lr)

        n_epochs = self.n_epochs
        batch_size = self.batch_size


        for epoch in range(n_epochs):
            for i in range(0, len(x_train), batch_size):
                Xbatch = x_train[i:i+batch_size]
                y_pred = self.forward(Xbatch)
                ybatch = y_train[i:i+batch_size]
                loss = loss_fn(y_pred, ybatch.reshape((batch_size, 1)))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            accuracy = None
            if x_val != None and y_val != None:
                with torch.no_grad():
                    y_pred = self(x_train)
        
                accuracy = (y_pred.round() == y_train).float().mean()

            print(f'Finished epoch {epoch+1}/{n_epochs}, loss {loss}', end="")
            if accuracy != None:
                print(f', accuracy {accuracy}')
            else:
                print()

    def predict_proba(self, x, y=None):
        with torch.no_grad():
            y_pred = self(x)
        
        if y != None:
            accuracy = (y_pred.round() == y).float().mean()
            print("accuracy: {accuracy}")

        return y_pred
    
    def save(self, path="our-model.pth"):
        # Save model
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path="our-model.pth"):
        # Create new model and load states
        newmodel = Model()
        newmodel.load_state_dict(torch.load(path))
        return newmodel
    
    def submit(self, test_x, path="submission.csv"):
        test_X = test_x[test_x.columns.difference(["patientunitstayid"])].to_numpy()
        test_X = torch.tensor(test_X, dtype=torch.float32)

        test_y_pred = self.predict_proba(test_X)

        submission = pd.DataFrame({"patientunitstayid":test_x["patientunitstayid"].to_numpy(), "hospitaldischargestatus":test_y_pred.reshape(-1)})
        submission.to_csv(path, header=True, index=False)
        print(f"Saved results to {path}")
