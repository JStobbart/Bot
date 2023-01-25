# Bot

Бот разработан [студентом](https://t.me/Satiata) Школы глубокого обучения МФТИ в рамках дипломного проекта "Телеграм-боты".

Бот осуществляет стилизацию изображений двумя способами:  
1 - Neural Transfer - перенос стиля с одного изображения на другое   
2 - StyleGAN - применение заготовленных стилей к изображению

Для режима Neural Transfer в качестве стиля рекомендуется использовать однородное изображение.  
Доступен выбор глубины переноса стиля:"

- 'не глубоко' - перенос стиля крайне поверхностный"
- 'средне' - рекомендуемая глубина переноса стиля, стиль просматривается на исходном изображении
- 'глубоко' - стиль преобладает над исходным изображением

В зависимости от выбранной глубины перенос занимает от 15 секунд до двух минут. При наличии GPU + CUDA скорость
возрастает.

Для режима CycleGAN доступны следующие стили:

- Monet
- Vangogh
- Cezanne
- ukiyoe (картины (образы) изменчивого мира) - направление в изобразительном искусстве Японии
- Aivazovsky (предобучен на [датасете](https://www.kaggle.com/competitions/painter-by-numbers/data) с Kaggle, логи и графики обучения расположены в директории [Aivazovsky_train_logs](https://github.com/JStobbart/Bot/tree/master/aivazovsky_train_logs))
- Лето -> Зима
- Зима -> Лето

Обработка изображения занимает около 10 секунд.

## Требования

- Linux or macOS
- Python 3
- Docker (не обязательно)
- NVIDIA GPU + CUDA (не обязательно)

## Начало работы

### Установка бота

- Клонируйте репозиторий и перейдите в папку бота:

```bash
git clone --recurse-submodules https://github.com/JStobbart/Bot.git
cd Bot
```


- Создайте и активируйте виртуальное окружение

```bash
python3 -m venv venv
source venv/bin/activate
```

- Установите [PyTorch](http://pytorch.org) и другие зависимости
    - без поддержки GPU (только CPU)
    ```bash
     pip install -r requirements_cpu.txt
     ```
    - с поддержкой GPU
     ```bash
     pip install -r requirements.txt
     ```

- Добавьте ваш [telegram bot token](https://core.telegram.org/bots#how-do-i-create-a-bot)
  в переменную окружения API_TOKEN

```bash
export API_TOKEN=ВАШ TELEGRAM BOT TOKEN ДОБАВЬТЕ СЮДА
```

- запустите бота

```bash
python3 main.py
```

### Установка бота для пользователей Docker

- Клонируйте репозиторий и перейдите в папку бота:

```bash
git clone --recurse-submodules https://github.com/JStobbart/Bot.git
cd Bot
```



#### Подготовка Dockerfile, сборка и запуск образа

- без поддержки GPU (только CPU)
    - откройте файл Dockerfile_cpu любым текстовым редактором и добавьте
      ваш [telegram bot token](https://core.telegram.org/bots#how-do-i-create-a-bot)
      в переменную окружения API_TOKEN между кавычками, как показано ниже:
        ```bash
        FROM python:3.9
        WORKDIR /app
        ENV API_TOKEN="ВАШ TELEGRAM BOT TOKEN ДОБАВЬТЕ СЮДА"
        ```
    - Создайте docker image
        ```bash
        docker build -f Dockerfile_cpu -t bot .
        ```
    - Запустите docker image
        ```bash
        docker run -d bot
        ```

- с поддержкой GPU
    - откройте файл Dockerfile_gpu любым текстовым редактором и добавьте
      ваш [telegram bot token](https://core.telegram.org/bots#how-do-i-create-a-bot)
      в переменную окружения API_TOKEN между кавычками, как показано ниже:

         ```bash
        FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
    
        WORKDIR /app
        ENV API_TOKEN="ВАШ TELEGRAM BOT TOKEN ДОБАВЬТЕ СЮДА"
            
        ```  
    - Создайте docker image

      ```bash
      docker build -f Dockerfile_gpu -t bot .
      ```
    - Запустите docker image

      ```bash
      docker run -gpus all -d bot
      ```

## При создании бота были использованы следующие материалы:
- статья [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- руководство [NEURAL TRANSFER USING PYTORCH](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
- репозиторий [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/)
