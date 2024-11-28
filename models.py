import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=2048, dropout_prob=0.3):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder with more layers and more filters
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 3x32x32 -> 64x16x16
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x16x16 -> 128x8x8
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 128x8x8 -> 256x4x4
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 256x4x4 -> 512x2x2
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # 512x2x2 -> 1024x1x1
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Flatten(),
            nn.Linear(1024, latent_dim),  # Increased latent dim
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        # Decoder with more layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),  # Increased latent dim
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Unflatten(1, (1024, 1, 1)),  # Reshape to 1024x1x1
            
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 1024x1x1 -> 512x2x2
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 512x2x2 -> 256x4x4
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 256x4x4 -> 128x8x8
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128x8x8 -> 64x16x16
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 64x16x16 -> 3x32x32
            # nn.Sigmoid()  # Use sigmoid to output normalized pixel values (0 to 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def generate(self, num_samples, bs=64, device=None):
        """
        Generate samples from the autoencoder by sampling from latent space.
        
        Args:
            num_samples (int): Number of samples to generate
            bs (int): Batch size for generation
            device (torch.device): Device to generate samples on
                
        Returns:
            torch.Tensor: Generated samples of shape (num_samples, C, H, W)
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Initialize list to store generated samples
        generated_samples = []
        
        # Calculate number of batches needed
        num_batches = (num_samples + bs - 1) // bs
        
        with torch.no_grad():
            for i in range(num_batches):
                # Calculate batch size for last batch
                current_bs = min(bs, num_samples - i * bs)
                
                # Sample from normal distribution in latent space
                z = torch.randn(current_bs, self.latent_dim, device=device)
                
                # Generate samples using decoder
                samples = self.decoder(z)
                
                # Append to list
                generated_samples.append(samples.cpu())
        
        # Concatenate all batches
        generated_samples = torch.cat(generated_samples, dim=0)
        
        # Ensure we return exactly num_samples
        return generated_samples[:num_samples]





# Variational Autoencoder (VAE)
class VAE(nn.Module):
    def __init__(self, latent_dim=2048, dropout_prob=0.3):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # Encoder with increased capacity
        self.encoder = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), # 3x32x32 -> 64x16x16
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 64x16x16 -> 128x8x8
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 128x8x8 -> 256x4x4
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # 256x4x4 -> 512x2x2
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1), # 512x2x2 -> 1024x1x1
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Flatten(),
        nn.Linear(1024, 1024), # Intermediate projection
        nn.ReLU(),
        nn.Dropout(dropout_prob)
        )
        # Separate layers for mean and log variance
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        # Decoder with matching capacity
        self.decoder = nn.Sequential(
        nn.Linear(latent_dim, 1024), # Match intermediate projection
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Unflatten(1, (1024, 1, 1)), # Reshape to 1024x1x1
        nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1), # 1024x1x1 -> 512x2x2
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 512x2x2 -> 256x4x4
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 256x4x4 -> 128x8x8
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 128x8x8 -> 64x16x16
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1), # 64x16x16 -> 3x32x32
        )


    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def generate(self, num_samples, bs=64, device=None):
        """
        Generate samples from the VAE by sampling from the latent distribution.
        
        Args:
            num_samples (int): Number of samples to generate
            bs (int): Batch size for generation
            device (torch.device): Device to generate samples on
            temperature (float): Temperature parameter for sampling (higher = more diverse)
                
        Returns:
            torch.Tensor: Generated samples of shape (num_samples, C, H, W)
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Initialize list to store generated samples
        generated_samples = []
        
        # Calculate number of batches needed
        num_batches = (num_samples + bs - 1) // bs
        
        with torch.no_grad():
            for i in range(num_batches):
                # Calculate batch size for last batch
                current_bs = min(bs, num_samples - i * bs)
                
                # Sample from standard normal distribution
                z = torch.randn(current_bs, self.latent_dim, device=device)
                
                # Generate samples using decoder
                samples = self.decoder(z)
                
                # Append to list
                generated_samples.append(samples.cpu())
        
        # Concatenate all batches
        generated_samples = torch.cat(generated_samples, dim=0)
        
        # Ensure we return exactly num_samples
        return generated_samples[:num_samples]

# GAN
import torch
import torch.nn as nn

# Convolutional Generator
import torch
import torch.nn as nn

# Refined Complex Convolutional Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=128):
        super(Generator, self).__init__()
        
        # Initial layer to upscale the latent vector
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),  # [batch_size, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
        # Upsample to 8x8
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # [batch_size, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        
        # Upsample to 16x16 with residual connection
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # [batch_size, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.residual2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)  # Residual connection
        
        # Upsample to 32x32 with another residual connection
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # [batch_size, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.residual3 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)  # Residual connection
        
        # Output layer to produce final RGB image
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),  # [batch_size, 3, 32, 32]
        )

    def forward(self, z):
        # Reshape the latent vector
        z = z.view(z.size(0), z.size(1), 1, 1)
        
        # Forward pass through the layers
        x = self.initial(z)
        x = self.upsample1(x)
        
        # Pass through the second upsampling and residual connection
        x = self.upsample2(x)
        x = x + self.residual2(x)  # Adding the residual connection
        
        # Pass through the third upsampling and residual connection
        x = self.upsample3(x)
        x = x + self.residual3(x)  # Adding the residual connection
        
        # Final output layer
        output = self.output_layer(x)
        return output


# Convolutional Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            # Input image: 3x32x32
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),  # [batch_size, 128, 16, 16]
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsample to 16x16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # [batch_size, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, padding=2, bias=False),  # [batch_size, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 512, 4, padding=2, bias=False),  # [batch_size, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 256, 4, padding=2, bias=False),  # [batch_size, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsample to 4x4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # [batch_size, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final convolution
            nn.Conv2d(512, 64, 4, 1, 0, bias=False),  # [batch_size, 64, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten to [batch_size, 64]
            nn.Linear(256, 256),  # First linear layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 256),  # Second linear layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1),  # Final classification layer
            nn.Sigmoid()  # Output between [0, 1]
        )

    # Combined forward pass
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)



class GANModelWrapper(nn.Module):
    def __init__(self, generator, discriminator, latent_dim=128):
        super(GANModelWrapper, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.train_step = 0  # Track whether to train generator or discriminator

    def forward(self, real_data):
        # Alternate between training discriminator and generator
        if self.train_step % 2 == 0:
            # Discriminator training step
            z = torch.randn(real_data.size(0), self.latent_dim, device=real_data.device)
            fake_data = self.generator(z).detach()  # Detach to avoid generator gradient update
            real_output = self.discriminator(real_data)
            fake_output = self.discriminator(fake_data)
            self.train_step += 1
            return real_output, fake_output, fake_data
        else:
            # Generator training step
            z = torch.randn(real_data.size(0), self.latent_dim, device=real_data.device)
            fake_data = self.generator(z)
            real_output = self.discriminator(real_data)
            fake_output = self.discriminator(fake_data)
            self.train_step += 1
            return fake_output, fake_data  # Only return fake output for generator loss calculation
    
    def generate(self, num_samples, bs=64, device=None):
        """
        Generate samples from the GAN using the generator network.
        
        Args:
            num_samples (int): Number of samples to generate
            bs (int): Batch size for generation
            device (torch.device): Device to generate samples on
            truncation (float, optional): Truncation value for latent sampling
                (if provided, clips latent vectors to improve sample quality)
                
        Returns:
            torch.Tensor: Generated samples of shape (num_samples, C, H, W)
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Initialize list to store generated samples
        generated_samples = []
        
        # Calculate number of batches needed
        num_batches = (num_samples + bs - 1) // bs
        
        with torch.no_grad():
            for i in range(num_batches):
                # Calculate batch size for last batch
                current_bs = min(bs, num_samples - i * bs)
                
                # Sample from latent space
                z = torch.randn(current_bs, self.latent_dim, device=device)
                
                # Apply truncation trick if specified
                # if truncation is not None:
                #     z = torch.clamp(z, -truncation, truncation)
                
                # Generate samples
                samples = self.generator(z)
                
                # Append to list
                generated_samples.append(samples.cpu())
        
        # Concatenate all batches
        generated_samples = torch.cat(generated_samples, dim=0)
        
        # Ensure we return exactly num_samples
        return generated_samples[:num_samples]


# Normalizing Flow (RealNVP)


class AffineCoupling(nn.Module):
    def __init__(self, dim, mask):
        super(AffineCoupling, self).__init__()
        self.register_buffer('mask', mask)
        self.masked_dim = int(mask.sum().item())
        self.unmasked_dim = dim - self.masked_dim
        
        # Simpler network with careful initialization
        self.net = nn.Sequential(
            nn.Linear(self.masked_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2 * self.unmasked_dim)
        )
        
        # Initialize last layer very close to zero
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # Get masked input
        masked_x = x * self.mask
        net_input = x[:, self.mask == 1]
        
        net_out = self.net(net_input)
        scale, translation = net_out.chunk(2, dim=1)
        
        # Very conservative scaling
        scale = torch.tanh(scale) * 1
        translation = torch.tanh(translation) *1
        
        # Create full tensors
        full_scale = torch.zeros_like(x)
        full_translation = torch.zeros_like(x)
        
        # Fill in the unmasked positions
        full_scale[:, self.mask == 0] = scale
        full_translation[:, self.mask == 0] = translation
        
        # Transform
        x_masked = x * (1 - self.mask)
        transformed = x_masked * torch.exp(full_scale) + full_translation * (1 - self.mask)
        y = masked_x + transformed
        
        return y, full_scale.sum(dim=1)

    def inverse(self, y):
        masked_y = y * self.mask
        net_input = y[:, self.mask == 1]
        
        net_out = self.net(net_input)
        scale, translation = net_out.chunk(2, dim=1)
        
        scale = torch.tanh(scale) * 1
        translation = torch.tanh(translation) *1
        
        full_scale = torch.zeros_like(y)
        full_translation = torch.zeros_like(y)
        
        full_scale[:, self.mask == 0] = scale
        full_translation[:, self.mask == 0] = translation
        
        y_masked = y * (1 - self.mask)
        x = (y_masked - full_translation * (1 - self.mask)) * torch.exp(-full_scale)
        x = masked_y + x
        
        return x
    
class RealNVP(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), num_coupling_layers=8):
        super(RealNVP, self).__init__()
        self.latent_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.input_shape = input_shape
        
        # Data normalization parameters with correct shapes
        self.register_buffer('data_mean', torch.zeros((1, *input_shape)))
        self.register_buffer('data_std', torch.ones((1, *input_shape)))
        
        # Initialize prior parameters
        self.register_buffer('prior_mean', torch.zeros(self.latent_dim))
        self.register_buffer('prior_std', torch.ones(self.latent_dim))
        
        # Create masks alternating between first and second half
        self.masks = []
        for i in range(num_coupling_layers):
            mask = torch.zeros(self.latent_dim, device='cuda')
            if i % 2 == 0:
                mask[:self.latent_dim//2] = 1
            else:
                mask[self.latent_dim//2:] = 1
            self.masks.append(mask)
        
        # Create coupling layers
        self.coupling_layers = nn.ModuleList([
            AffineCoupling(self.latent_dim, self.masks[i])
            for i in range(num_coupling_layers)
        ])

    @property
    def prior(self):
        """Get the prior distribution"""
        return Normal(self.prior_mean, self.prior_std)

    def forward(self, x):
        # Normalize input
        x_normalized = (x - self.data_mean) / self.data_std
        
        # Add small noise during training for better sampling
        if self.training:
            x_normalized = x_normalized + torch.randn_like(x_normalized) * 0.01
        
        batch_size = x_normalized.size(0)
        z = x_normalized.view(batch_size, -1)
        log_det = torch.zeros(batch_size, device=x.device)
        
        for layer in self.coupling_layers:
            z, layer_log_det = layer(z)
            log_det += layer_log_det
            
        return z, log_det

    def inverse(self, z, denormalize=True):
        x = z
        for layer in reversed(self.coupling_layers):
            x = layer.inverse(x)
        x = x.view(-1, *self.input_shape)
        
        if denormalize:
            x = x * self.data_std + self.data_mean
        return x

    def sample(self, num_samples, temperature=0.7):
        # Sample from prior
        z = self.prior.sample((num_samples,))
        if temperature != 1.0:
            z = z * temperature
        return self.inverse(z)

def update_normalization(model, dataloader):
    """Update normalization statistics for the model"""
    means = []
    stds = []
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(model.data_mean.device)
            means.append(data.mean(dim=0, keepdim=True))  # Keep batch dimension
            stds.append(data.std(dim=0, keepdim=True))
    
    mean = torch.stack(means).mean(dim=0)  # Average over batches
    std = torch.stack(stds).mean(dim=0)
    
    # Copy with correct shapes
    model.data_mean.copy_(mean)
    model.data_std.copy_(std)
    

class RealNVPWrapper(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), num_coupling_layers=8):
        super(RealNVPWrapper, self).__init__()
        self.model = RealNVP(input_shape, num_coupling_layers)

    def forward(self, x):
        # Get latent representation and log determinant
        z, log_det = self.model(x)
        
        # Generate samples
        with torch.no_grad():
            generated_samples = self.model.sample(5, temperature=0.7)
            
        # Get reconstruction
        reconstruction = self.model.inverse(z)
        
        return generated_samples, reconstruction, z, log_det
    def __init__(self, input_shape=(3, 32, 32), num_coupling_layers=8):
        super(RealNVPWrapper, self).__init__()
        self.model = RealNVP(input_shape, num_coupling_layers)

    def forward(self, x):
        # Get latent representation and log determinant
        z, log_det = self.model(x)
        
        # Generate samples
        with torch.no_grad():
            generated_samples = self.model.sample(5, temperature=0.7)
            
        # Get reconstruction
        reconstruction = self.model.inverse(z)
        
        return generated_samples, reconstruction, z, log_det
    
    def generate(self, num_samples, bs=64, device=None):
        """
        Generate samples from the RealNVP model.
        
        Args:
            num_samples (int): Number of samples to generate
            bs (int): Batch size for generation
            device (torch.device): Device to generate samples on
            temperature (float): Temperature for sampling (controls diversity)
                
        Returns:
            torch.Tensor: Generated samples of shape (num_samples, C, H, W)
        """
        if device is None:
            device = next(self.parameters()).device
            
        generated_samples = []
        num_batches = (num_samples + bs - 1) // bs
        
        with torch.no_grad():
            for i in range(num_batches):
                current_bs = min(bs, num_samples - i * bs)
                
                # Generate samples using inverse flow
                samples = self.model.sample(current_bs, temperature=0.7)
                generated_samples.append(samples.cpu())
        
        # Concatenate and return exact number of samples
        generated_samples = torch.cat(generated_samples, dim=0)
        return generated_samples[:num_samples]