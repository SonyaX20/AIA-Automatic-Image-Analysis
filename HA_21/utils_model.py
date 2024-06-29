import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
import seaborn as sns
from time import time

# models

class ClassificationNet(nn.Module):
    def __init__(self):
        """
        Classification Net for task 1
        -------
        conv1: 1*28*28 -> 32*28*28
        pool: 32*28*28 -> 32*14*14
        conv2: 32*14*14 -> 64*14*14
        pool: -> 64*7*7
        fc1: -> 128
        fc2: -> 10
        output_size = (input_size - kernel_size + 2 * padding) / stride +1
        output_size = input_size
        """
        super(ClassificationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7) # -1 automatic batch size
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# functions

def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    """
    general training procedure
    """
    model.train() # initialize to training
    start_time = time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device) # load in cuda
            optimizer.zero_grad() # clear grad accumulation in torch
            outputs = model(inputs) # feed forward
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() # .item(): read scaler from tensor
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}") # average loss over iterations,
    end_time = time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")