
import os

import torch
from PIL import Image
import torchvision.transforms as transforms

from net.devices import get_device

device = get_device()


def get_image_size():
    imsize = 512 if torch.cuda.is_available() else 256  # use small size if no gpu
    return imsize


def get_loader(imsize):
    #imsize = get_image_size()
    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
      #  transforms.CenterCrop(imsize),  # centercrop with size
        transforms.ToTensor()])  # transform it into a torch tensor
    return loader


def image_loader(image_name, imsize):

    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# def image_loader(image_name, shape):
#
#     loader = get_loader(shape[2:])
#
#     image = Image.open(image_name)
#     # fake batch dimension required to fit network's input dimensions
#     image = loader(image).unsqueeze(0)
#     return image.to(device, torch.float)


def get_content_and_style(content, style):
    # content_img = image_loader(content)  # content
    imsize = get_image_size()
    content_img = image_loader(content, imsize)
    shape = content_img.shape[2:]
    style_img = image_loader(style, shape)  # style

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"
    return content_img, style_img


def saved(tensor, title='test.jpg'):
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    image.save(title)


async def delete_pict(file, directory=False):
    if directory:
        try:
            os.rmdir(file)
        except:
            return False
    else:
        try:
            os.remove(file)
            return True
        except Exception:
            return False


def get_img_gan(img, title):
    image = Image.open(img)
    loader = transforms.Compose([
        transforms.Resize(256),  # scale imported image
        transforms.CenterCrop(256),  # centercrop with size
        transforms.ToTensor()])  # transform it into a torch tensor
    image = loader(image).unsqueeze(0)
    saved(image, title)
    return title
