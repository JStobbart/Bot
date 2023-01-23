import torch


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def gpu(flag=False):
    if flag:
        if torch.cuda.is_available():
            return 512
        else:
            return 256
    else:
        return 0 if torch.cuda.is_available() else -1