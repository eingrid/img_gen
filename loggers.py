import torch 
import numpy as np

class BASIC_LOGGER():
    def __init__(self, writer):
        self.writer = writer
        self.notitication = 1
        self.additional_metrics = {}

    def log_loss(self,metric_name, avg_loss, step):
        self.writer.add_scalar(metric_name, avg_loss, step)

    def additional_logs(self,outputs,smth_else):
        if self.notitication != 0:
            print("additional_logs are not defined for this logger")
            self.notitication -= 1 
    
    def plot_additional_logs(self,epoch):
        if self.notitication != 0:
            print("plot_additional_logs are not defined for this logger")
            self.notitication -= 1 
    
        

class VAE_LOGGER(BASIC_LOGGER):
    # def set_samples(self,num_samples=5):
    #     self.num_samples = num_samples

    def add_images(self,n, output, data, epoch, dataformats):
        indices = np.random.choice(len(data), 5, replace=False)
        random_samples = data[indices]
        random_reconstructed = output[0][indices]
        
        # Concatenate originals and reconstructions vertically (stacking)
        comparison = torch.cat([random_samples, random_reconstructed], dim=0)  # Concatenate originals and reconstructions
        
        # all_epoch_images.append(comparison)

        self.writer.add_images(f'Reconstruction/All_Epochs', comparison, epoch, dataformats=dataformats)

class AE_LOGGER(BASIC_LOGGER):

    def add_images(self,n, output, data, epoch, dataformats):
        indices = np.random.choice(len(data), 5, replace=False)
        random_samples = data[indices]
        random_reconstructed = output[indices]
        
        # Concatenate originals and reconstructions vertically (stacking)
        comparison = torch.cat([random_samples, random_reconstructed], dim=0)  # Concatenate originals and reconstructions
        
        # all_epoch_images.append(comparison)

        self.writer.add_images(f'Reconstruction/All_Epochs', comparison, epoch, dataformats=dataformats)

class GAN_LOGGER(BASIC_LOGGER):
    def add_images(self,n, output, data, epoch, dataformats='NCHW'):
        """
        Logs real vs. fake images for GANs, or original vs. reconstructed images for AEs/VAEs.

        Args:
            output: Output from the model, which could be:
                    - (real_output, fake_output) for GANs (discriminator step)
                    - fake_output for GANs (generator step)
                    - reconstructed data for AEs/VAEs
            data: Real data batch, either for GANs or AEs/VAEs.
            epoch: Current epoch number.
            dataformats: The format of the image data, typically 'NCHW' (channels, height, width).
        """
        # If output is a tuple, assume it’s from the discriminator step (real_output, fake_output)
        if isinstance(output, tuple) and len(output) == 3:
            real_output, fake_output, fake_data = output
        else:
            fake_output, fake_data = output
        
        # print(fake_data)
        # print(fake_data.shape)
        num_images = min(5, len(data))
        
        # Randomly select a few images from real and fake outputs
        indices = np.random.choice(len(data), num_images, replace=False)
        real_samples = data[indices]
        fake_samples = fake_data[indices].detach()  # Detach to avoid unwanted gradient calculations
        
        # Ensure fake_samples are in the same shape as real_samples (4D)
        if fake_samples.dim() == 2:  # If fake_samples is 2D, reshape to 4D (assume grayscale)
            fake_samples = fake_samples.unsqueeze(2).unsqueeze(3)  # Reshape to [batch_size, channels, 1, 1]
            fake_samples = fake_samples.expand(-1, -1, real_samples.size(2), real_samples.size(3))  # Expand to match height/width
        elif fake_samples.dim() == 3:  # If fake_samples is 3D (channels, height, width), add batch dimension
            fake_samples = fake_samples.unsqueeze(0).expand(real_samples.size(0), -1, -1, -1)  # Expand to match batch size
        
        # Concatenate real and fake images along vertical axis for comparison
        comparison = torch.cat([real_samples, fake_samples], dim=0)
        self.writer.add_images(f'GAN/Real_vs_Fake_Epoch', comparison, epoch, dataformats=dataformats)
    

    def additional_logs(self, outputs,step):
        if "correct_classified_fakes" not in self.additional_metrics:
            self.additional_metrics["correct_classified_fakes"] = 0
            self.additional_metrics["correct_classified_real"] = 0
            self.additional_metrics["total_shape"] = 0
            

        if isinstance(outputs,tuple) and len(outputs) == 3:
            real_output, fake_output, fake_data = outputs
            self.additional_metrics["correct_classified_fakes"] += (fake_output < 0.5).sum()
            self.additional_metrics["correct_classified_real"] += (real_output >= 0.5).sum()
            self.additional_metrics["total_shape"] += len(fake_output) + len(real_output)

    
    def plot_additional_logs(self,step):
        total_correct = self.additional_metrics["correct_classified_fakes"] \
              + self.additional_metrics["correct_classified_real"] \
              
        accuracy = total_correct/self.additional_metrics['total_shape'] 
        self.log_loss("Discriminator Accuracy",accuracy,step)

        self.additional_metrics["correct_classified_fakes"] = 0
        self.additional_metrics["correct_classified_real"] = 0
        self.additional_metrics["total_shape"] = 0

        

class NF_LOGGER(BASIC_LOGGER):
    def add_images(self,n, output, data, epoch, dataformats):
        indices = np.random.choice(len(data), 5, replace=False)
        random_samples = data[indices]
        random_reconstructed = output[1][indices]#.view(-1, 3, 32, 32)
        # Concatenate originals and reconstructions vertically (stacking)
        generated = output[0]#.view(-1, 3, 32, 32)
        comparison = torch.cat([random_samples, random_reconstructed,generated], dim=0)  # Concatenate originals and reconstructions
        print(random_samples.device, random_reconstructed.device, generated.device)
        print(comparison.device)

        print(comparison.shape)        
        # all_epoch_images.append(comparison)

        self.writer.add_images(f'Reconstruction/All_Epochs', comparison, epoch, dataformats=dataformats)