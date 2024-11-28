import torch
import torch.nn as nn
import numpy as np

class RealNVPLoss(nn.Module):
    def __init__(self, prior, device):
        super(RealNVPLoss, self).__init__()
        self.prior = prior
        self.device = device
        # Move prior parameters to the specified device
        self.prior.loc = self.prior.loc.to(device)
        self.prior.scale = self.prior.scale.to(device)

    def forward(self, model_output, data):
        """Compute loss in bits per dimension"""
        _,_,z,log_det = model_output
        
        # Ensure z is not too extreme
        z = torch.clamp(z, -10, 10)
        
        # Calculate dimensions (should be 3072 for CIFAR)
        batch_size = z.size(0)
        dims = np.prod(data.shape[1:])  # Should be 3*32*32 = 3072
        
        # Calculate log probability of z under the prior
        log_prob_z = self.prior.log_prob(z).sum(dim=1)  # Sum over dimensions
        
        # Scale the log_det properly
        log_det = log_det / dims  # Scale by dimensions
        
        # Proper scaling from [0, 255] to [0, 1]
        log_transform = -np.log(256)  # Just the per-dimension scaling
        
        # Total loss in bits per dimension
        nll_loss = -(log_prob_z/dims + log_det + log_transform)
        
        # Debug prints
        print(f"Log prob z (per dim): {(log_prob_z/dims).mean():.2f}")
        print(f"Log det (per dim): {log_det.mean():.2f}")
        print(f"Log transform (per dim): {log_transform:.2f}")
        print(f"Loss (bits per dim): {nll_loss.mean():.2f}")
        
        return nll_loss.mean()
