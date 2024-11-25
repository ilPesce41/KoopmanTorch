from torch import nn

class BaseLoss(nn.Module):
    def __init__(self, name: str):
        super(BaseLoss, self).__init__()
        self.name = name

    def compute_loss(self, x, y) -> float:
        raise NotImplementedError

    def forward(self, x, y):
        loss = self.compute_loss(x, y)
        return loss

    def __call__(self, x, y):
        return self.forward(x, y)
    
class ReconstructionLoss(BaseLoss):
    def __init__(self):
        super(ReconstructionLoss, self).__init__('reconstruction')

    def compute_loss(self, x, y) -> float:
        return nn.functional.mse_loss(x, y)
    
class PredictionLoss(BaseLoss):
    def __init__(self):
        super(PredictionLoss, self).__init__('prediction')

    def compute_loss(self, x, y) -> float:
        return nn.functional.mse_loss(x, y)
    
class LinearityLoss(BaseLoss):
    def __init__(self):
        super(LinearityLoss, self).__init__('linearity')

    def compute_loss(self, x, y) -> float:
        return nn.functional.mse_loss(x, y)