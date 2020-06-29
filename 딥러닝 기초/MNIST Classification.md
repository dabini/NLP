# MNIST Classification

### 1. models.py

```python
import torch
import torch.nn as nn

class ImageClassifier(nn.Module):
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        super().__init__()
        
        self.layers = nn.Sequential(
        	nn.Linear(input_size, 500),
        	nn.LeakyReLU(),
        	nn.BatchNormld(500),
        	nn.Linear(500, 400),
        	nn.LeakyReLU(),
        	nn.BatchNormld(400),
        	nn.Linear(400, 300),
        	nn.LeakyReLU(),
        	nn.BatchNormld(200),
        	nn.Linear(200, 100),
        	nn.LeakyReLU(),
        	nn.BatchNormld(100),
        	nn.Linear(100, 50),
        	nn.LeakyReLU(),
        	nn.BatchNormld(50),
        	nn.Linear(50, output_size),
        	nn.Softmax(dim=-1),
        )
        
    def forward(self, x):
        # |x| = (batch_size, input_size)
        
        y = self.layers(x)
        # |y| = (batch_size, output_size)
        
        return y
```



### 2. trainer.py

```python
from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

class Trainer():
    
    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit
        
        super().__init__()
        
    def _train(self, x, y, config):
        self.model.train()
        
        # Shuffle before begin
        indices = torch.randperm(x, size(0), deivce=x.device)
        x = torch.index_select(x, dim=0, index=indeces).split(config.batch_size, dim=0)
        y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)
        
        total_loss = 0
        
        for i, (x_i, y_i) in enumerate(zip(x,y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat, y_i.squeeze())
            
            # Initialize the gradients of the model
            self.optimizer.zero_grad()
            loss_i.backward()
            
            self.optimizer.step()
            
            if config.verbose >= 2:
                print("TRAIN INTERATION(%d/%d): loss=%.4e" % (i+1, len(x), float(loss_i)))
                
            # Don't forget to detach to prevent memory leak
            total_loss += float(loss_i)
        return total_loss / len(x)
    
    def _validate(self, x, y, config):
        # Turn evaluation mode on
        self.model.eval()
        
        # Turn on the no_grad mode to make more efficinity
        with torch.no_grad():
            # Suffle before begin
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indeces).split(config.batch_size, dim=0)
        	y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)
        
        	total_loss = 0
        
        	for i, (x_i, y_i) in enumerate(zip(x,y)):
            	y_hat_i = self.model(x_i)
            	loss_i = self.crit(y_hat, y_i.squeeze())
            
            	if config.verbose >= 2:
                	print("TRAIN INTERATION(%d/%d): loss=%.4e" % (i+1, len(x), float(loss_i)))
                
            	total_loss += float(loss_i)
                
        	return total_loss / len(x)
        
    
    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None
        
        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self.validate(valid_data[0], valid_data[1], config)
            
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())
            
            print("Epoch(%d/%d): train_loss = %.4e valid_loss=%.4e lowest_loss=%.4e" % (
            	epoch_index +1 ,
            	config.n_epochs,
            	train_loss,
            	valid_loss,
            	lowest_loss)
            )
        
        # Restore to best model.
        self.model.load_state_dict(best_model)
```

