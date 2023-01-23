import gc

from torchvision import models
import torch
from net.preparePicture import saved, get_content_and_style
import torch.nn as nn
import torch.optim as optim
from net.loss import ContentLoss, StyleLoss


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class NeuralTransferNet:
    """
    Инкапсулированный вариант нейросети по копированию стиля.
    За основу взят туториал https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    Оригинальные комментарии сохранены.
    За пределы класса вынесены:
    - функции потерь и матрица Грама
    - функции предобработки изображений
    - функция, определяющая доступность использования видеокарты для вычислений
    """
    def __init__(self):
        """
        при инициализации объекта класса создается предобученная vgg19 и ряд переменных, необходимых для работы,
        например, номера слоев используемых для расчета потерь взяты в соовтетствии со статьей https://arxiv.org/abs/1508.06576

        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn = models.vgg19(weights='DEFAULT').features.to(self.device).eval()

        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.content_layers_default = ['conv_16']
        self.style_layers_default = ['conv_1', 'conv_3', 'conv_5', 'conv_9', 'conv_13']

    def __del__(self):
        print("Start destructor...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Destructed")

    def start(self, content, style, num_steps, title):
        """
        Функция запускает нейронную сеть и реализовывает принцип инкапсуляции
        Возвращает путь до готового изображения
        """
        return self.run_transfer(self.cnn, content, style, num_steps, title)

    def run_transfer(self, cnn, content, style, num_steps, title='test.jpg'):
        content_img, style_img = get_content_and_style(content, style)
        input_img = content_img.clone()
        output = self.run_style_transfer(cnn, self.cnn_normalization_mean, self.cnn_normalization_std,
                                         content_img, style_img, input_img, num_steps=num_steps, style_weight=1000000,
                                         content_weight=1)
        saved(output, title=title)
        return title

    def run_style_transfer(self, cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=300,
                           style_weight=1000000, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses(cnn,
                                                                              normalization_mean, normalization_std,
                                                                              style_img, content_img,
                                                                              content_layers=self.content_layers_default,
                                                                              style_layers=self.style_layers_default)

        # We want to optimize the input and not the model parameters so we
        # update all the requires_grad fields accordingly
        input_img.requires_grad_(True)
        model.requires_grad_(False)

        optimizer = self.get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score  # alpha & betta
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img

    def get_style_model_and_losses(self, cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers,
                                   style_layers):
        # normalization module
        normalization = Normalization(normalization_mean, normalization_std).to(self.device)

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        #  print(f"model: {model}")

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        #  print(f"model_after: {model}")
        #  print(f"content_losses: {content_losses}, \n style_losses: {style_losses}")

        return model, style_losses, content_losses

    def get_input_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img])
        return optimizer
