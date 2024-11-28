import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import numpy as np
from scipy import linalg
from typing import Tuple, Optional, Union

class SmallImageFeatureExtractor(nn.Module):
    """Feature extractor for small images like CIFAR"""
    def __init__(self):
        super(SmallImageFeatureExtractor, self).__init__()
        # Use ResNet18 instead of Inception
        resnet = models.resnet18(pretrained=True)
        # Remove final fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.eval()
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Resize images to 64x64 for better feature extraction
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        features = self.feature_extractor(x)
        return features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions

class FIDMetric:
    def __init__(self, device: Optional[torch.device] = None) -> None:
        """
        Initialize FID metric calculator for small images.
        
        Args:
            device: Device to run calculations on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = SmallImageFeatureExtractor().to(self.device)
        self.feature_extractor.eval()

    def _calculate_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate mean and covariance statistics of features."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def _calculate_frechet_distance(self, mu1: np.ndarray, sigma1: np.ndarray,
                                  mu2: np.ndarray, sigma2: np.ndarray,
                                  eps: float = 1e-6) -> float:
        """Calculate Fréchet distance between two multivariate Gaussians."""
        covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        
        return float(
            np.sum((mu1 - mu2) ** 2) + 
            np.trace(sigma1) + 
            np.trace(sigma2) - 
            2 * tr_covmean
        )

    @torch.no_grad()
    def _get_features(self, images: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        """Extract features from images."""
        features = []
        n_batches = (len(images) + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            batch = images[start_idx:end_idx].to(self.device)
            
            # Ensure values are in range [0, 1]
            if batch.min() < 0 or batch.max() > 1:
                batch = (batch + 1) / 2  # Convert from [-1, 1] to [0, 1]
            
            batch_features = self.feature_extractor(batch)
            features.append(batch_features.cpu().numpy())
            
        return np.concatenate(features, axis=0)

    def __call__(self,
                 real_images: torch.Tensor,
                 generated_images: torch.Tensor,
                 batch_size: int = 32) -> float:
        """
        Calculate FID score between real and generated images.
        
        Args:
            real_images: Tensor of real images (N, C, H, W)
            generated_images: Tensor of generated images (N, C, H, W)
            batch_size: Batch size for feature extraction
            
        Returns:
            FID score
        """
        # Validate inputs
        if real_images.shape != generated_images.shape:
            raise ValueError(f"Image batches must have same shape. Got {real_images.shape} and {generated_images.shape}")
        
        if real_images.shape[1] != 3:
            raise ValueError(f"Expected RGB images (3 channels), got {real_images.shape[1]} channels")
        
        # Extract features
        real_features = self._get_features(real_images, batch_size)
        gen_features = self._get_features(generated_images, batch_size)
        
        # Calculate statistics
        mu_real, sigma_real = self._calculate_statistics(real_features)
        mu_gen, sigma_gen = self._calculate_statistics(gen_features)
        
        # Calculate FID
        fid_score = self._calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        
        return fid_score
