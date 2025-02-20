import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# a simple MLP
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=512, output_dim=1):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        if type(hidden_dim) == int:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc1half = nn.Linear(hidden_dim, hidden_dim)
            self.fc1half2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        elif type(hidden_dim) == list:
            self.fc1 = nn.Linear(input_dim, hidden_dim[0])
            self.fc1half = nn.Linear(hidden_dim[0], hidden_dim[1])
            self.fc1half2 = nn.Linear(hidden_dim[1], hidden_dim[2])
            self.fc2 = nn.Linear(hidden_dim[2], output_dim, bias=False)


    def embd(self, x):
        x = torch.nn.GELU()(self.fc1(x))
        x = torch.nn.GELU()(self.fc1half(x))
        x = torch.nn.GELU()(self.fc1half2(x))
        return x

    def lastlayer(self, x):
        return self.fc2(x)[:,0]

    def forward(self, x):
        x = self.fc2(self.embd(x))
        return x[:,0]

class simpleMLP(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim = 64, output_dim = 1):
        super(simpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim, bias=False)

    def embd(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def forward(self, x):
        x = self.fc4(self.embd(x))
        return x


class BradleyTerry(nn.Module):
    def __init__(self, scorenet = simpleMLP()):
        super(BradleyTerry, self).__init__()
        self.scorenet = scorenet
    def forward(self, team1, team2):
        team1_strength = self.scorenet(team1)
        team2_strength = self.scorenet(team2)
        return torch.sigmoid(team1_strength - team2_strength)

def BTloss(pred, target):
    #breakpoint()
    return F.binary_cross_entropy(pred, 1.*target)



def three_well_fun(team,scale=5):
    x = team[:,0]
    y = team[:,1]
    return scale * (3*torch.exp(-x**2 - (y-(1.0/3)) **2) - 3*torch.exp( -x**2 - (y-(5.0/3))**2) - 5 * torch.exp( -(x-1)**2-y**2) - 5*torch.exp(-(x+1)**2-y**2 ) + 0.2*x**4 + 0.2*(y-(1.0/3))**4)


def simulate_label(team1, team2, score_model = three_well_fun):
    pred = score_model(team1) -  score_model(team2)
    return torch.bernoulli( torch.sigmoid( pred)).float()


def train_BT(team1, team2, target, score_model=None, lr = 0.001, niter = 1000):
    model = BradleyTerry(scorenet=score_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(niter):
        pred = model(team1.to(device), team2.to(device))
        loss = BTloss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(loss)
    return model.scorenet

def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x))
    return ranks

def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x.cpu())
    y_rank = _get_ranks(y.cpu())

    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)

def spearman_correlation_batch(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (500, N, )
        y: Shape (N, )
    """
    total_outs = []
    import numpy as np
    for idx_ in range(len(x)):
        x_rank = _get_ranks(x[idx_].cpu())
        y_rank = _get_ranks(y[idx_].cpu())
        n = x[idx_].size(0)
        upper = 6 * torch.sum((x_rank - y_rank).pow(2))
        down = n * (n ** 2 - 1.0)
        total_outs.append(1.0 - (upper / down))
    return np.mean(total_outs)

def get_batch_bon(x: torch.Tensor, y: torch.Tensor):
    total_outs = []
    import numpy as np
    for idx_ in range(len(x)):
        sub_bon_now = []
        for bon_n in [5, 10, 30, 50, 100, 300, 500]:
            rand_sample_idx = torch.randperm(len(x[idx_]))[:bon_n]
            max_idx_of_predict = torch.argmax(x[idx_][rand_sample_idx])
            bon_now = y[idx_][rand_sample_idx][max_idx_of_predict]
            sub_bon_now.append(bon_now)
        total_outs.append(sub_bon_now)
    return np.mean(total_outs, 0)
