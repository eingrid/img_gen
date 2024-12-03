import torch.nn as nn
import torch.optim as optim
import torch
from mse_loss import CustomBCELoss

class GANOptimizer:
    def __init__(self, generator, discriminator, lr=0.0002, betas=(0.5, 0.999)):
        self.generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
        self.discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
        self.train_step = 0

    def zero_grad(self):
        if self.train_step % 2 == 0:
            self.discriminator_optimizer.zero_grad()
        else:
            self.generator_optimizer.zero_grad()

    def step(self):
        if self.train_step % 2 == 0:
            self.discriminator_optimizer.step()
        else:
            self.generator_optimizer.step()
        self.train_step += 1



class GANCriterion:
    def __init__(self, adversarial_loss=CustomBCELoss()):
        self.adversarial_loss = adversarial_loss

    def __call__(self, output, target=None):
        if isinstance(output, tuple) and len(output) == 3:
            # Discriminator loss
            real_output, fake_output, fake_data = output
            real_target = torch.ones_like(real_output)
            fake_target = torch.zeros_like(fake_output)
            real_loss = self.adversarial_loss(real_output, real_target)
            fake_loss = self.adversarial_loss(fake_output, fake_target)
            return real_loss + fake_loss
        else:
            fake_output, fake_data = output
            # Generator loss
            target = torch.ones_like(fake_output)  # Generator wants fake data classified as real
            return self.adversarial_loss(fake_output, target)

