import time

import torch

from net.devices import get_device
from net.preparePicture import saved, get_content_and_style
from net.transfer_net import run_style_transfer

device = get_device()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# переделать на асинхрон
def run_transfer(cnn, content, style, num_steps, title='test.jpg'):
    content_img, style_img = get_content_and_style(content, style)
    input_img = content_img.clone()
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, num_steps=num_steps, style_weight=1000000, content_weight=1)
    saved(output, title=title)
    return title
