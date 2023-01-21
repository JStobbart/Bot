# Bot


## Prerequisites
- Linux or macOS
- Python 3
- Docker (optional)
- NVIDIA GPU + CUDA (optional)

## Getting Started
### Installation for Docker users
- Clone this repo:
```bash
git clone https://github.com/JStobbart/Bot.git
cd Bot
```
- Inside Bot folder clone official StyleGAN repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
```

- Open Dockerfile by your text redactor and add your [telegram bot token](https://core.telegram.org/bots#how-do-i-create-a-bot) 
in the API_TOKEN environment variable:
```
FROM python:3.9

WORKDIR /home
ENV API_TOKEN="YOUR API_TOKEN" <-- here

```
- make docker image
```bash
docker build -t bot .
```
- run docker image
```bash
docker run -d bot
```

### No Docker installation

- Clone this repo:
```bash
git clone https://github.com/JStobbart/Bot.git
cd Bot
```
- Inside Bot folder clone official StyleGAN repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
```

- create environment (recommended)
```bash
python3 -m venv venv
# then activate
source venv/bit/activate
```
- Install [PyTorch](http://pytorch.org) and other dependencies (e.g., aiogram, torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
```bash
pip install -r requirements.txt
```
- add your [telegram bot token](https://core.telegram.org/bots#how-do-i-create-a-bot) 
in the API_TOKEN environment variable
```bash
export API_TOKEN=your telegram bot token
```

  
