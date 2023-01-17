from torchvision import models


def get_cnn(device):
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    return cnn





