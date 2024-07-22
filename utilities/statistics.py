import torch

from typing import Union

def autocorrelation(tensor:torch.Tensor, dims:Union[int,tuple[int,...]], mean_dim:int, split:int=1):
    if isinstance(dims, int): dims = (dims, )

    tensor = tensor.cpu()

    L = 1.0E-18
    a = tensor - torch.mean(tensor, dim=mean_dim, keepdim=True)
    a = a / (torch.sqrt(torch.sum(a**2, dim=mean_dim, keepdim=True)/(a.size(mean_dim) - 1)) + L)
    
    paddings  = [0 for i in range(2*len(a.size()))]
    paddings_ = [0 for i in range(2*len(a.size()))]
    multiplier = 1.0
    for dim in dims:
        paddings[2*dim]      = (a.size(dim) + 1)//2
        paddings[2*dim + 1]  = paddings[2*dim]
        paddings_[2*dim]     = -paddings[2*dim]
        paddings_[2*dim + 1] = -paddings[2*dim + 1]
        multiplier *= a.size(dim)

    try:
        result = []
        for data in torch.chunk(a, split, dim=mean_dim):
            data = data.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            data = torch.nn.functional.pad(data, paddings)
            spectrum    = torch.fft.fftshift(torch.fft.fftn(data, dim=dims))
            convolution = torch.fft.ifftshift(torch.fft.ifftn(spectrum*spectrum.conj(), dim=dims)).abs()
            convolution = torch.nn.functional.pad(convolution, paddings_)
            result.append(convolution.cpu())
        convolution = torch.cat(result, dim=mean_dim)
        result = torch.sum(convolution, dim=mean_dim) / (a.size(mean_dim) - 1)
        return result
    except torch.cuda.OutOfMemoryError as error:
        if a.size(mean_dim) >= split*2:
            print(f'Out of Memory Error, split:{split*2}')
            return autocorrelation(tensor, dims, mean_dim, split*2)
        else:
            print(f'Out of Memory Error, resizing')
            for dim in dims:
                tensor = tensor.swapdims(0, dim)
                tensor = tensor[::2]
                tensor = tensor.swapdims(0, dim)
            return autocorrelation(tensor, dims, mean_dim, split)



def correlation_circle(correlation:torch.Tensor, limits:tuple[tuple[float,float],...]=None, percent:float=0.7):
    if limits is None: limits = tuple([(0, correlation.size(i)) for i in range(len(correlation.size()))])
    if len(limits) != len(correlation.size()): raise ValueError

    with torch.no_grad():
        if hasattr(torch, 'unravel_index'):
            center_index = torch.unravel_index(correlation.argmax(), correlation.shape)
        else:
            flattened_index = correlation.argmax()
            center_index = (flattened_index // correlation.shape[1], flattened_index % correlation.shape[1])
        arrays = [torch.linspace(limit0, limit1, correlation.size(i)) for i, (limit0, limit1) in enumerate(limits)]
        centers = [array[index].item() for array, index in zip(arrays, center_index)]
        arrays = [array - array[index] for array, index in zip(arrays, center_index)]
        r_mesh = torch.zeros(correlation.size())
        for mesh in torch.meshgrid(*arrays,indexing='ij'):
            r_mesh += mesh**2
        integral = torch.sum(correlation) * (limits[0][1] - limits[0][0]) * (limits[1][1] - limits[1][0])
        maximum = torch.max(correlation)
    def function(r:float) -> torch.Tensor:
        mask = r_mesh < r**2
        return torch.min(torch.where(mask, correlation, maximum)) / maximum < percent

    radius0 = 0
    radius1 = 2*max([limit1 - limit0 for limit0, limit1 in limits])
    state0 = function(radius0)
    state1 = function(radius1)
    if state0 == state1: return radius1, centers

    while abs(radius0 - radius1) > 1.0E-12:
        radius = (radius0 + radius1)/2
        state = function(radius)
        if state == state0:
            radius0 = radius
            state0 = state
        else:
            radius1 = radius
            state1 = state
    radius = (radius0 + radius1)/2

    return radius, centers

def distribution(a:torch.Tensor, N:int=100, return_values:bool=False):
    values = torch.linspace(a.min(), a.max(), N+1)[1:]
    results = torch.zeros(N, dtype=torch.float32, device=a.device)
    for i, value in enumerate(values):
        results[i] = torch.sum(a <= value)
    results /= a.numel()
    if return_values:
        return results, values
    return results