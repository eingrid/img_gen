import torch
import torch.nn.functional as F
from mse_loss import CustomMSELoss

class VAE_Loss(torch.nn.Module):
    def __init__(self):
        super(VAE_Loss, self).__init__()
        self.mse_loss = CustomMSELoss()

    def forward(self, reconstructed_mu_var : tuple, original):
        reconstructed, mu, log_var = reconstructed_mu_var
        batch_size = reconstructed.size(0)
        # Reconstruction loss (mean squared error) / BCE
        reconstruction_loss = self.mse_loss(reconstructed, original)  # Sum over the batch
        # print(batch_size)
        # KL Divergence loss
        # Formula: 0.5 * (mu^2 + exp(log_var) - log_var - 1)
        kl_divergence = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()),axis=0)
        kl_loss = kl_divergence 
        # Total VAE loss
        print("KL", kl_loss)
        # print("KL scaled", kl_loss * 1e-3)
        print("REC", reconstruction_loss)
        total_loss = reconstruction_loss + kl_loss  * 1e-5
        # print(total_loss)
        return total_loss