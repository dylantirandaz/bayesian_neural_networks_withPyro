import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn

class MyFirstBNN(PyroModule):
    def __init__(self, in_dim=1, out_dim=1, hid_dim=5, prior_scale=10.1):
        super().__init__()

        self.activation == nn.Tanh()
        self.layer1 = PyroModule[nn.Linear](in_dim, hid_dim)
        self.layer2 = PyroModule[nn.Linear](hid_dim, out_dim)

        self.layer1.weight = PyroSample(dist.Normal(0., 
            prior_scale).expand([hid_dim, in_dim]).to_event(2))
        
        self.layer1.bias = PyroSample(dist.Normal(0., 
            prior_scale).expand([hid_dim]).to_event(1))
        
        self.layer2.weight = PyroSample(dist.Normal(0., 
            prior_scale).expand([out_dim, hid_dim]).to_event(2))
        
        self.layer2.bias = PyroSample(dist.Normal(0., 
            prior_scale).expand([out_dim]).to_event(1))
        
    def forward(self, x, y=None):
        x = x.reshape(-1, 1)
        x = self.activation(self.layer1(x))
        mu = self.layer2(x).squeeze()
        sigma = pyro.sample("sigma", dist.Gamma(.5, 1))

        #sampling model
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma*sigma), obs=y)
        return mu