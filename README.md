# Bot
Бот разработан студенотом Школы глубокого обучения МФТИ в рамках дипломного проекта "Телеграм-боты".

Бот осуществляет стилизацию изображений двумя способами:  
1 - Neural Transfer - перенос стиля с одного изображения на другое  
2 - StyleGAN - применение заготовленных стилей к изображению

Для режима Neural Transfer в качестве стиля рекомендуется использовать однородное изображение.  
Доступен выбор глубины переноса стиля:"
- 'не глубоко' - перенос стиля крайне поверхностный"
- 'средне' - рекомендуемая глубина переноса стиля, стиль просматривается на исходном изображении
- 'глубоко' - стиль преобладает над исходным изображением

В зависимости от выбранной глубины перенос занимает от 15 секунд до двух минут. При наличии GPU + CUDA скорость возрастает.

Для режима StyleGAN доступны следующие стили:
- Monet
- Vangogh
- Cezanne
- ukiyoe (картины (образы) изменчивого мира) - направление в изобразительном искусстве Японии
- Лето -> Зима
- Зима -> Лето

Обработка изображения занимает около 10 секунд.

## Требования
- Linux or macOS
- Python 3
- Docker (не обязательно)
- NVIDIA GPU + CUDA (не обязательно)

## Начало работы
### Установка

- Клонируйте репозиторий и перейдите в папку бота:
```bash
git clone https://github.com/JStobbart/Bot.git
cd Bot
```
- Клонируйте официальный репозиторий [CycleGAN in pytorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix):
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
```

- Создайте виртуальное окружение (не обязательно)
```bash
python3 -m venv venv
# then activate
source venv/bit/activate
```
- Установите [PyTorch](http://pytorch.org) и другие зависимости
```bash
pip install -r requirements.txt
```
- Добавьте ваш [telegram bot token](https://core.telegram.org/bots#how-do-i-create-a-bot) 
в переменную окружения API_TOKEN
```bash
export API_TOKEN=your telegram bot token
```
- запустите бота
```bash
python main.py
```

### Установка с использованием Docker
- Клонируйте репозиторий и перейдите в папку бота:
```bash
git clone https://github.com/JStobbart/Bot.git
cd Bot
```
- Клонируйте официальный репозиторий [CycleGAN in pytorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix):
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
```

- Откройте Dockerfile любым текстовым редактором и добавьте ваш [telegram bot token](https://core.telegram.org/bots#how-do-i-create-a-bot) 
в переменную окружения API_TOKEN между кавычками, как показано ниже:
```
FROM python:3.9

WORKDIR /home
ENV API_TOKEN="ВАШ API_TOKEN ДОБАВЬТЕ СЮДА"

```
- Создайте docker image
```bash
docker build -t bot .
```
- Запустите docker image
```bash
docker run -d bot
```


