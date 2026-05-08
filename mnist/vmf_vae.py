
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from collections import defaultdict
from tqdm import tqdm
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
        )
    def forward(self, x):
        return x + self.net(x)

class ConvModelVAE(nn.Module):
    def __init__(self, z_dim, distribution="vmf"):
        super().__init__()

        self.z_dim = z_dim
        self.distribution = distribution

        # Encoder: 1x28x28 -> 64x7x7
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.SiLU(),
        )

        if distribution == "normal":
            self.fc_mean = nn.Linear(256, z_dim)
            self.fc_var = nn.Linear(256, z_dim)
        elif distribution == "vmf":
            self.fc_mean = nn.Linear(256, z_dim)
            self.fc_var = nn.Linear(256, 1)
        else:
            raise NotImplementedError


        self.fc_decode = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 64 * 7 * 7),
            nn.SiLU(),
        )

        # Decoder: z -> 1x28x28 logits
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            ResBlock(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            ResBlock(32),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )
    def encode(self, x):
        h = self.encoder(x)

        if self.distribution == "normal":
            z_mean = self.fc_mean(h)
            z_var = F.softplus(self.fc_var(h)) + 1e-5

        elif self.distribution == "vmf":
            z_mean = self.fc_mean(h)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            z_var = F.softplus(self.fc_var(h)) + 1

        return z_mean, z_var

    def decode(self, z):
        h = self.fc_decode(z)
        x_logits = self.decoder(h)
        return x_logits

    def reparameterize(self, z_mean, z_var):
        if self.distribution == "normal":
            q_z = torch.distributions.Normal(z_mean, z_var)
            p_z = torch.distributions.Normal(
                torch.zeros_like(z_mean),
                torch.ones_like(z_var),
            )

        elif self.distribution == "vmf":
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim - 1, device=z_mean.device)

        return q_z, p_z

    def forward(self, x):
        z_mean, z_var = self.encode(x)
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        x_logits = self.decode(z)

        return (z_mean, z_var), (q_z, p_z), z, x_logits

class ModelVAE(torch.nn.Module):
    
    def __init__(self, h_dim, z_dim, activation=F.relu, distribution='normal'):
        """
        ModelVAE initializer
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        super(ModelVAE, self).__init__()
        
        self.z_dim, self.activation, self.distribution = z_dim, activation, distribution
        
        # 2 hidden layers encoder
        self.fc_e0 = nn.Linear(784, h_dim * 2)
        self.fc_e1 = nn.Linear(h_dim * 2, h_dim)

        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var =  nn.Linear(h_dim, z_dim)
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, 1)
        else:
            raise NotImplemented
            
        # 2 hidden layers decoder
        self.fc_d0 = nn.Linear(z_dim, h_dim)
        self.fc_d1 = nn.Linear(h_dim, h_dim * 2)
        self.fc_logits = nn.Linear(h_dim * 2, 784)

    def encode(self, x):
        # 2 hidden layers encoder
        x = self.activation(self.fc_e0(x))
        x = self.activation(self.fc_e1(x))
        
        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            z_mean = self.fc_mean(x)
            z_var = F.softplus(self.fc_var(x))
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            z_mean = self.fc_mean(x)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(self.fc_var(x)) + 1
        else:
            raise NotImplemented
        
        return z_mean, z_var
        
    def decode(self, z):
        
        x = self.activation(self.fc_d0(z))
        x = self.activation(self.fc_d1(x))
        x = self.fc_logits(x)
        
        return x
        
    def reparameterize(self, z_mean, z_var):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        elif self.distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim - 1, device=z_mean.device)
        else:
            raise NotImplemented

        return q_z, p_z
        
    def forward(self, x): 
        z_mean, z_var = self.encode(x)
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        x_ = self.decode(z)
        
        return (z_mean, z_var), (q_z, p_z), z, x_
    
