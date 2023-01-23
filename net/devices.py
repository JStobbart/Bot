import torch


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def gpu():
    return 0 if torch.cuda.is_available() else -1
