import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import time

class Model(nn.Module):

    def __init__(self, n_features=85, lr=0.00001, n_epochs=100, batch_size=10):
        super().__init__()
        self.hidden1 = nn.Linear(n_features, 128)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(128, 50)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(50, 24)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(24, 1)
        self.act_output = nn.Sigmoid()

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
        X = torch.tensor(x_train.to_numpy(), dtype=torch.float32)
        y = torch.tensor(y_train.to_numpy(), dtype=torch.float32)

        have_validation_data = False
        if isinstance(x_val, pd.DataFrame) and isinstance(y_val, pd.Series):
            val_X = torch.tensor(x_val.to_numpy(), dtype=torch.float32)
            val_y = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
            have_validation_data = True

        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=float(self.lr))

        n_epochs = self.n_epochs
        batch_size = self.batch_size

        # useful for when len(X) % batch_size != 0
        def get_batch(t, start, batch_size):
            if (start + batch_size) > len(t):
                return t[start:]
            else:
                return t[start:start+batch_size]

        start_time = time.time()
        for epoch in range(n_epochs):
            for i in range(0, len(X), batch_size):
                Xbatch = get_batch(X, i, batch_size)
                y_pred = self.forward(Xbatch)
                ybatch = get_batch(y, i, batch_size)
                loss = loss_fn(y_pred, ybatch.reshape((-1, 1)))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f'Epoch {epoch+1}/{n_epochs}: loss {loss:.3f}', end="")
            accuracy = None
            if have_validation_data:
                with torch.no_grad():
                    y_pred = self(val_X)
                accuracy = (y_pred.round() == val_y).float().mean()
                print(f", accuracy {accuracy:.3f}")
            else:
                with torch.no_grad():
                    y_pred = self(X)
                accuracy = (y_pred.round() == y).float().mean()
                print(f", accuracy {accuracy:.3f}")
        
        end_time = time.time()
        print(f"Training completed in {(end_time - start_time):.1f} seconds.")

    def predict_proba(self, x, y=None):
        with torch.no_grad():
            y_pred = self(x)
        
        if y != None:
            accuracy = (y_pred.round() == y).float().mean()
            print(f"accuracy: {accuracy}")

        return y_pred
    
    def save(self, path="our-model.pth"):
        torch.save(self.state_dict(), path)
        print(f"Saved model to {path}")

    @staticmethod
    def load(path="our-model.pth"):
        print(f"Loading model from {path}")
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
