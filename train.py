import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data_loader import load_cifar10_data  # Importing the data loading function
import numpy as np
import os

def train(model, train_loader, criterion, optimizer, device, epoch, logger, log_interval=100):
    model.train()  # Set model to training mode
    running_loss = 0.0
    num_batches = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [Train]")):
        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)

        #gan case
        # if isinstance(loss,tuple) and len(loss) == 3:
        # Backward pass and optimization
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        num_batches += 1
        optimizer.step()
        
        running_loss += loss.item()
        
    avg_loss = running_loss / num_batches
    step = epoch 
    logger.log_loss('Loss/train', avg_loss, step)



def validate(model, val_loader, criterion, device, epoch, logger, metric):
    """
    Validation function for Autoencoder/VAE models, with image reconstruction logging.
    Args:
        model: The model being validated.
        val_loader: DataLoader for the validation dataset.
        criterion: Loss function, typically MSE for Autoencoders/VAEs.
        device: Device to run the validation on (CPU or GPU).
        epoch: Current epoch number.
        logger: Logger for metrics and images.
        metric: FID metric calculator.
    Returns:
        Average validation loss.
    """
    model.eval()
    val_loss = 0.0
    num_batches = len(val_loader)
    required_samples = 512
    
    # For collecting real images
    real_images = []
    collected_samples = 0
    
    with torch.no_grad():
        # First collect required number of real images
        for data, _ in tqdm(val_loader, desc=f"Epoch {epoch} [Validation]"):
            # Calculate how many more samples we need
            samples_needed = min(data.size(0), required_samples - collected_samples)
            
            if samples_needed > 0:
                # Select and store samples
                real_batch = data[:samples_needed].to(device)
                real_images.append(real_batch)
                collected_samples += samples_needed
            
            # Process batch for validation metrics
            data = data.to(device)
            output = model(data)
            logger.additional_logs(output, epoch)
            # Compute loss and accumulate
            loss = criterion(output, data)
            val_loss += loss.item()
            
            # If we have collected enough real images, stop collecting
            if collected_samples >= required_samples:
                # Combine all real images
                real_images = torch.cat(real_images, dim=0)[:required_samples]
                break
        
        # Generate samples and compute FID
        generated_samples = model.generate(required_samples, 64, device)
        fid_score = metric(real_images, generated_samples)
        
        # Calculate average validation loss
        avg_val_loss = val_loss / num_batches
        
        # Log metrics
        logger.log_loss('Loss/validation', avg_val_loss, epoch)
        logger.log_loss('FID_score/validation', fid_score, epoch)
        logger.plot_additional_logs(epoch)
        logger.add_images(f'Reconstruction/All_Epochs', output, data, epoch, dataformats='NCHW')
        
        return avg_val_loss, fid_score


def save_checkpoint(model, path="saved_model/best_model.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
