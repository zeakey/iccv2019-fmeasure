from torch import nn

class FLoss(nn.Module):
    def __init__(self, beta=0.3, log_like=False):
        super(FLoss, self).__init__()
        self.beta = beta
        self.log_like = log_like

    def forward(self, prediction, target):
        EPS = 1e-10
        N = prediction.size(0)
        TP = (prediction * target).view(N, -1).sum(dim=1)
        H = self.beta * target.view(N, -1).sum(dim=1) + prediction.view(N, -1).sum(dim=1)
        fmeasure = (1 + self.beta) * TP / (H + EPS)
        if self.log_like:
            floss = -torch.log(fmeasure)
        else:
            floss  = (1 - fmeasure)
        return floss

def floss(prediction, target, beta=0.3, log_like=False):
    EPS = 1e-10
    N = N = prediction.size(0)
    TP = (prediction * target).view(N, -1).sum(dim=1)
    H = beta * target.view(N, -1).sum(dim=1) + prediction.view(N, -1).sum(dim=1)
    fmeasure = (1 + beta) * TP / (H + EPS)
    if log_like:
        floss = -torch.log(fmeasure)
    else:
        floss  = (1 - fmeasure)
    return floss


if __name__=="__main__":
    import torch
    fl = FLoss()
    prediction = torch.rand(1, 1, 100, 100)
    target = (torch.rand(1, 1, 100, 100) >= 0.5).float()
    print("FLoss (Module)=%.7f" % fl(prediction, target).item())
    print("FLoss (Functional)=%.7f" % floss(prediction, target).item())