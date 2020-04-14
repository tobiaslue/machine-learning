import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, 50)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(50, 100)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(100, output_size)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.lin1(x))
        x = self.relu2(self.lin2(x))
        out = self.out_act(self.out(x))
        return out

class Classifier():
    def __init__(self, input_size, output_size, plot_loss=False):
        super().__init__()
        self.net = Net(input_size, output_size)
        self.opt = optim.Adam(self.net.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        self.plot_loss = plot_loss

    def _train_epoch(self, X, Y, batch_size=50):
        self.net.train()
        losses = []
        for beg_i in range(0, X.size(0), batch_size):
            x_batch = X[beg_i:beg_i + batch_size, :]
            y_batch = Y[beg_i:beg_i + batch_size, :]
            x_batch = Variable(x_batch)
            y_batch = Variable(y_batch)

            self.opt.zero_grad()
            # (1) Forward
            y_hat = self.net(x_batch)
            # (2) Compute diff
            loss = self.criterion(y_hat, y_batch)
            # (3) Compute gradients
            loss.backward()
            # (4) update weights
            self.opt.step()        
            losses.append(loss.data.numpy())
        return losses

    def train(self, X, y):
        e_losses = []
        num_epochs = 15
        for e in range(num_epochs):
            e_losses += self._train_epoch(X, y)
        
        if self.plot_loss:
            plt.plot(e_losses)
            plt.show()

    def predict(self, X):
        self.net.eval()
        return self.net(X)


    


