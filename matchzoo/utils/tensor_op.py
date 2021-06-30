import torch


def load_tensors_to(tensors, device):
    if isinstance(tensors, list):
        for i, t in enumerate(tensors):
            if not isinstance(t, torch.Tensor):
                continue
            tensors[i] = t.to(device)
    elif isinstance(tensors, dict):
        for k, t in tensors.items():
            if not isinstance(t, torch.Tensor):
                continue
            tensors[k] = t.to(device)
    elif isinstance(tensors, torch.Tensor):
        tensors = tensors.to(device)
    else:
        raise ValueError('Input must be a torch.Tensor or \
            a container of torch.Tensor, but the input type is %s.' % type(tensors))

    return tensors
