import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomBCELoss(nn.Module):
    """
    Custom Binary Cross Entropy Loss implementation for PyTorch.
    
    Args:
        reduction (str): Specifies the reduction method:
            'none': no reduction
            'mean': mean of the losses
            'sum': sum of the losses
            Default: 'mean'
        weight (torch.Tensor, optional): Manual rescaling weight for each batch element.
            If provided, has to be a Tensor of size (N,)
        pos_weight (torch.Tensor, optional): Weight for positive examples.
            Must be a vector with length equal to number of classes.
        eps (float): Small constant for numerical stability. Default: 1e-8
    """
    
    def __init__(self, reduction='mean', weight=None, pos_weight=None, eps=1e-8):
        super(CustomBCELoss, self).__init__()
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.eps = eps
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss calculation.
        
        Args:
            predictions (torch.Tensor): Predicted probabilities (0 to 1)
            targets (torch.Tensor): Ground truth binary labels (0 or 1)
            
        Returns:
            torch.Tensor: Computed BCE loss
            
        Raises:
            ValueError: If shapes don't match or if prediction values are outside [0,1]
        """
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions shape {predictions.shape} "
                           f"!= targets shape {targets.shape}")
        
        predictions = torch.clamp(predictions, min=self.eps, max=1 - self.eps)
        
        if self.pos_weight is not None:
            loss = self.pos_weight * targets * torch.log(predictions) + \
                   (1 - targets) * torch.log(1 - predictions)
        else:
            loss = targets * torch.log(predictions) + \
                   (1 - targets) * torch.log(1 - predictions)
        
        # Apply sample weights if provided
        if self.weight is not None:
            if self.weight.shape[0] != predictions.shape[0]:
                raise ValueError(f"Weight tensor size {self.weight.shape[0]} "
                               f"!= batch size {predictions.shape[0]}")
            loss = loss * self.weight.view(-1, 1)
        
        # Negative of the average
        loss = -loss
        
        # Apply reduction method
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}")
    

class CustomMSELoss(nn.Module):
    """
    Custom Mean Squared Error Loss implementation for PyTorch.
    This implementation allows for optional averaging across batch dimension
    and custom weighting of samples.
    
    Args:
        reduction (str): Specifies the reduction method to apply to the output:
            'none': no reduction will be applied
            'mean': the sum of the output will be divided by the number of elements
            'sum': the output will be summed
            Default: 'mean'
        weight (torch.Tensor, optional): a manual rescaling weight given to each
            sample. If given, it has to be a Tensor of size (N,) where N is the
            batch size. Default: None
    """
    
    def __init__(self, reduction='mean', weight=None):
        super(CustomMSELoss, self).__init__()
        self.reduction = reduction
        self.weight = weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss calculation.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth values
            
        Returns:
            torch.Tensor: Computed loss value
            
        Raises:
            ValueError: If predictions and targets have different shapes
            ValueError: If weight tensor shape doesn't match batch size
        """
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions shape {predictions.shape} "
                           f"!= targets shape {targets.shape}")
        
        squared_diff = (predictions - targets) ** 2
        
        # Apply sample weights if provided
        if self.weight is not None:
            if self.weight.shape[0] != predictions.shape[0]:
                raise ValueError(f"Weight tensor size {self.weight.shape[0]} "
                               f"!= batch size {predictions.shape[0]}")
            squared_diff = squared_diff * self.weight.view(-1, 1)
        
        # Apply reduction method
        if self.reduction == 'none':
            return squared_diff
        elif self.reduction == 'sum':
            return torch.sum(squared_diff)
        elif self.reduction == 'mean':
            return torch.mean(squared_diff)
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}")