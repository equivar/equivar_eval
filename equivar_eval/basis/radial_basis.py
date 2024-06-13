import torch

class GaussianBasisProjection(torch.nn.Module):
    '''Basis of translated Gaussians
    '''
    def __init__(
            self,
            start:float=0.0,
            stop:float=5.0,
            num_gaussians:int=50
            ):
        super().__init__()
        offset=torch.linspace(start,stop,num_gaussians)
        self.register_buffer('offset',offset)
        self.gamma=-0.5/(offset[1]-offset[0]).item()**2

    def forward(self,r:torch.Tensor):
        r=r.view(-1,1)-self.offset.view(1,-1)
        return torch.exp(self.gamma*torch.pow(r,2))


class GaussianCosEnvelopeBasisProjection(torch.nn.Module):
    '''Translated Gaussians multiplied by a cosine envelope (cutoff) function
    '''
    def __init__(
            self,
            start:float=0.0,
            stop:float=5.0,
            num_gaussians:int=50
            ):
        super().__init__()
        offset=torch.linspace(start,stop,num_gaussians)
        self.register_buffer('offset',offset)
        self.alpha=torch.pi/stop
        self.gamma=-0.5/(offset[1]-offset[0]).item()**2

    def forward(self,r:torch.Tensor):
        rcol=r.view(-1,1)
        roffset=rcol-self.offset.view(1,-1)
        return 0.5*(1.0+torch.cos(self.alpha*rcol))*torch.exp(self.gamma*torch.pow(roffset,2))
