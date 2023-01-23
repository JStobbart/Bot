import os
import logging

from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text

from keyboards import inline_keyboard_confirm, inline_keyboard_gan, start_buttons, stop_button

from states import Transform, Gan

from net import delete_pict, get_img_gan, gpu
from net.net_class import NeuralTransferNet

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=os.environ.get('API_TOKEN'))
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
pict_dir_name = 'data'


async def run_style_transfer(content, style, num_steps):
    """
    Функция служит связующим звеном между ботом и нейросетью по копированию стиля (NeuralTransferNet).
    Функция принимает пути до исходного изображения и изображения стиля, а также число эпох.
    Создает экземпляр класса нейросети по копированию стиля изображения.
    Передает полученные значения и параметр title (путь и наименование будущего стилизованного изображения)
    в метод старт, далее происходит работа сети.
    Функция возвращает тотже title (путь до сгенерированного изображения)
    """

    net = NeuralTransferNet()
    out = net.start(content, style, num_steps, title=f'{content[:-4]}_styled.jpg')
    return out


async def cycle_gan_connector(id_chat, style, content):
    """функция служит связующим звеном между ботом и нейросетью CycleGAN.
    Формирование запроса для CycleGAN:
    dataroot - директория с изображением пользователя
    name - наименование предобученной модели (находятся в папке checkpoints/pretrained_models/)
    model - режим модели test, т.к. мы загружаем предобученную модель
    no_droput - дропауты не нужны, т.к. тест режим
    gpu_ids - функция gpu() автоматически определяет доступна ли видеокарта для вычислений (torch.cuda.is_available())
    results_dir - директория куда будет сохранено итоговое изображение
    load_size, crop_size - обработка для размера изображения 512x512

    Далее происходит вызов CycleGAN через сформированный запрос в консоль
    В переменную created_path присваивается путь до обработанного изображения и возвращается боту
    """
    path = f'python pytorch-CycleGAN-and-pix2pix/test.py ' \
           f'--dataroot ./data{id_chat} ' \
           f'--name pretrained_models/{style} ' \
           f'--model test --no_dropout ' \
           f'--gpu_ids {gpu()} ' \
           f'--results_dir ./result/ ' \
           f'--load_size 512 ' \
           f'--crop_size 512'
    os.system(path) # запуск CycleGAN с указанными выше параметрами
    # т.к. мы знаем путь по которому будет сгенерировано изображение, присвоем его переменной created_path
    created_path = f"./result/pretrained_models/{style}/test_latest/images/{content[:-4]}_fake.png"
    return created_path


###########################
"""Neural transfer part"""
###########################


@dp.message_handler(text='Neural transfer')
async def ntn(message: types.Message):
    """хэндлер срабатывает при выборе Neural transfer в меню бота
    (либо при написании этого текста в чат боту)"""

    await message.answer(text='Выбран режим: Neural transfer', reply_markup=types.ReplyKeyboardRemove())
    await message.answer(f'Отправьте мне изображение, которое хотите обработать', reply_markup=stop_button)
    await Transform.content.set()  # готовимся ловить исходное изображение в переменную content, объекта класса Transform


@dp.message_handler(state=Transform.content, content_types=types.ContentType.PHOTO)
async def get_content(message: types.Message, state: FSMContext):
    """
    Хэндлер ловит исходное изображение, предназначенное для перменной content объекта класса Transform (state)
    При этом, хэндлер сработает только на изображение (параметр content_types)
    """

    await state.update_data(id_chat=message.chat.id) # дополнительно ловим id пользователя
    pict_dir = f"{pict_dir_name}{message.chat.id}"  # нужно для создания отдельных папок для каждого пользователя. Вид - ./dataUSERIDNUMBER/
    content_picture = f"{pict_dir}/{message.photo[-1].file_id}.jpg" # путь для сохранения изображения
    await message.photo[-1].download(destination_file=f"{content_picture}") # сохранение изображения

    await state.update_data(content=content_picture) # полученное изображение (точнее путь до него) передаем в переменную content
    await message.answer('Отправьте мне изображение с которого хотите скопировать стиль')
    await Transform.style.set() # готовимся ловить исходное изображение в переменную style, объекта класса Transform


@dp.message_handler(state=Transform.style, content_types=types.ContentType.PHOTO)
async def get_style(message: types.Message, state: FSMContext):
    pict_dir = f"{pict_dir_name}{message.chat.id}" # нужно для создания отдельных папок для каждого пользователя. Вид - ./dataUSERIDNUMBER/
    style_picture = f"{pict_dir}/{message.photo[-1].file_id}.jpg" # путь для сохранения изображения
    await message.photo[-1].download(destination_file=f"{style_picture}") # сохранение изображения стиля

    await state.update_data(style=style_picture) # полученное изображение (точнее путь до него) передаем в переменную content

    # Далее выводится inline клавиатура с выбором глубины обучения
    await message.answer('Выберите глубину копирования стиля. Чем глубже копирование стиля, тем дольше будет идти обработка', reply_markup=inline_keyboard_confirm)


@dp.callback_query_handler(text=['50epochs', '300epochs', '600epochs'], state='*')
async def agree(call: types.CallbackQuery, state: FSMContext):
    """
    Хэндлер ловит ответ на инлайн клавиатуру с выбором глубины обучения


    """
    data = await state.get_data() # достаем сохраненные переменные с путями к изображениям и id пользователя
    content = data.get('content')
    style = data.get('style')
    id_chat = data.get('id_chat')

    # далее в зависимости от выбранной глубины будут инициализорованы переменные с определенными величинами
    if call.data == '50epochs':
        epochs = 50
        deep = 'не глубокое'
    elif call.data == '300epochs':
        epochs = 300
        deep = 'среднее'
    elif call.data == '600epochs':
        epochs = 600
        deep = 'глубокое'
    else:
        await state.finish() # останавливаем машину состояний
        # удаляем исходные изображения
        for file in [content, style]:
            await delete_pict(f"{file}")

        await call.message.answer(f"/start for another picture")
        return

    await state.finish()  # останавливаем машину состояний, т.к. все перменные пойманы
    await call.message.answer(f"Вы выбрали: {deep} копирование стиля. Пожалуйста, ожидайте...")

    """to neural transfer
    далее вызываем наш "коннектор" к нейросети, который возвращает путь 
    к обработанному изображению в переменную out
    """
    out = await run_style_transfer(content, style, num_steps=epochs)

    pict = types.InputFile(path_or_bytesio=f"{out}")
    await call.message.answer(f"Готово")
    # отправляем изображение пользователю
    await call.message.answer_photo(photo=pict)

    # удаляем исходные изображения и итоговое изображение
    for file in [content, style, out]:
        await delete_pict(f"{file}")
    await delete_pict(f"{pict_dir_name}{id_chat}", directory=True)

    await call.message.answer(text='Выберите действие: ', reply_markup=start_buttons)


##################
"""GAN part"""
##################


@dp.message_handler(text='CycleGAN')
async def ntn(message: types.Message):
    """хэндлер срабатывает при выборе CycleGAN в меню бота
        (либо при написании этого текста в чат боту)"""

    await message.answer(text='Выбран режим: CycleGAN', reply_markup=stop_button)
    await message.answer('Отправьте мне изображение, которое хотите обработать')
    await Gan.content_gan.set()  # готовимся ловить исходное изображение в переменную content, объекта класса Gan


@dp.message_handler(state=Gan.content_gan, content_types=types.ContentType.PHOTO)
async def get_content(message: types.Message, state: FSMContext):
    await state.update_data(id_chat_gan=message.chat.id)  # ловим id пользователя
    pict_dir = f"{pict_dir_name}{message.chat.id}"  # нужно для создания отдельных папок для каждого пользователя. Вид - ./dataUSERIDNUMBER/

    content_picture = f"{pict_dir}/{message.photo[-1].file_id}.jpg"  # путь для сохранения изображения
    await message.photo[-1].download(destination_file=f"{content_picture}")  # сохранение изображения

    get_img_gan(content_picture, title=content_picture)  # подготовка изображения для CycleGAN

    await state.update_data(content_gan=f"{message.photo[-1].file_id}.jpg") # наименование исходного изображения записываем в перменную content

    await Gan.style_gan.set() # готовимся ловить стиль в переменную style
    await message.answer('Выбери стиль', reply_markup=inline_keyboard_gan)


@dp.callback_query_handler(text=['style_monet_pretrained', 'style_vangogh_pretrained', 'style_cezanne_pretrained', 'style_ukiyoe_pretrained',
                                 'style_aivazovsky_pretrained', 'summer2winter_yosemite_pretrained', 'winter2summer_yosemite_pretrained'],
                           state=Gan.style_gan)
async def style_gan(call: types.CallbackQuery, state: FSMContext):
    await state.update_data(style_gan=call.data)  # записываем стиль в переменную style

    data = await state.get_data()  # достаем сохраненные переменные
    content = data.get('content_gan')
    style = data.get('style_gan')
    id_chat = data.get('id_chat_gan')

    await call.message.answer(f"Применяю стиль к исходному изображению...")

    # Передаем путь к исходному изображению, выбранный стиль и ID пользователя в функцию-коннектор с CycleGAN
    created_path = await cycle_gan_connector(id_chat, style, content)
    # коннектор вернул путь к сгенерированному изображению
    pict = types.InputFile(path_or_bytesio=created_path) # загружаем сгенерированное изображение
    await call.message.answer(f"Готово")
    await call.message.answer_photo(photo=pict)  # отправляем изображение

    await state.finish()  # останавливаем машину состояний

    # удаляем изображения и директорию пользователя
    await delete_pict(f"{pict_dir_name}{id_chat}/{content}")
    await delete_pict(f"{created_path}")
    await delete_pict(f"{created_path[:-8]}real.png")
    await delete_pict(f"{pict_dir_name}{id_chat}", directory=True)

    await call.message.answer(text='Выберите действие: ', reply_markup=start_buttons)


@dp.callback_query_handler(text='cancel_gan_code', state='*')
async def cancel(call: types.CallbackQuery, state: FSMContext):
    """
    функция сброса выбора стиля CycleGAN и машины состояния
    с удалением изображения и директории пользователя
    """
    data = await state.get_data()
    content = data.get('content_gan')
    await state.finish()
    await delete_pict(f"{content}")
    await delete_pict(f"data{call.message.chat.id}", directory=True)
    await call.message.answer(text='Выберите действие: ', reply_markup=start_buttons)


@dp.callback_query_handler(text='cancel_code', state='*')
async def cancel(call: types.CallbackQuery, state: FSMContext):
    """
    функция сброса выбора глубины обучения Neural Transfer Net
    и машины состояний с удалением изображений и директории пользователя
    """
    data = await state.get_data()
    content = data.get('content')
    style = data.get('style')
    await state.finish()
    await delete_pict(f"{content}")
    await delete_pict(f"{style}")
    await delete_pict(f"data{call.message.chat.id}", directory=True)
    await call.message.answer(text='Выберите действие: ', reply_markup=start_buttons)


@dp.message_handler(Text(equals=['stop', '/stop', 'cancel', '/cancel'], ignore_case=True), state='*')
async def cancel(message: types.Message, state: FSMContext):
    """
    функция остановки и сброса машины состояний
    """
    await state.finish()
    await delete_pict(f"data{message.chat.id}", directory=True)
    await message.answer('Выберите действие: ', reply_markup=start_buttons)


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.answer(text=f'Привет, {message.chat.username}!\n'
                              f'Бот разработан студенотом Школы глубокого обучения МФТИ в рамках дипломного проекта "Телеграм-боты"'
                              'Бот осуществляет стилизацию изображений двумя способами:\n'
                              '1 - Neural Transfer - перенос стиля с одного изображения на другое\n'
                              '2 - CycleGAN - применение заготовленных стилей к изображению\n'
                              '/help - подробное описание работы бота')
    await message.answer(text='Выберите действие: ', reply_markup=start_buttons)


@dp.message_handler(Text(equals=['help', '/help'], ignore_case=True))
async def send_welcome(message: types.Message):

    await message.answer(text='Бот поддерживает два режима работы:\n'
                              '1 - Neural Transfer - перенос стиля с одного изображения на другое\n'
                              '2 - CycleGAN - применение заготовленных стилей к изображению')
    await message.answer(text=f"Для режима Neural Transfer в качестве стиля рекомендуется использовать однородное изображение.\n"
                              f"Также доступен выбор глубины переноса стиля:\n"
                              f"- 'не глубоко' - перенос стиля крайне поверхностный\n"
                              f"- 'средне' - рекомендуемая глубина переноса стиля, стиль просматривается на исходном изображении\n"
                              f"- 'глубоко' - стиль преобладает над исходным изображением\n"
                              f"В зависимости от выбранной глубины перенос занимает от 15 секунд до двух минут.")
    await message.answer(text=f"Для режима CycleGAN доступны следующие стили:\n"
                              f"- Monet\n"
                              f"- Vangogh\n"
                              f"- Cezanne\n"
                              f"- ukiyoe (картины (образы) изменчивого мира) - направление в изобразительном искусстве Японии\n"
                              f"- Aivazovsky\n"
                              f"- Лето -> Зима\n"
                              f"- Зима -> Лето\n"
                              f"Обработка изображения занимает около 10 секунд")
    await message.answer(text='Выберите действие: ', reply_markup=start_buttons)


@dp.message_handler()
async def echo(message: types.Message):

    await message.answer(f"/start для начала работы")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
