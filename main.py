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
    net = NeuralTransferNet()
    out = net.start(content, style, num_steps, title=f'{content[:-4]}_styled.jpg')
    return out


###########################
"""Neural transfer part"""
###########################


@dp.message_handler(text='Neural transfer')
async def ntn(message: types.Message):
    await message.answer(text='Выбран режим: Neural transfer', reply_markup=types.ReplyKeyboardRemove())
    await message.answer(f'Отправьте мне изображение, которое хотите обработать', reply_markup=stop_button)
    await Transform.content.set()


@dp.message_handler(state=Transform.content, content_types=[types.ContentType.PHOTO, types.ContentType.DOCUMENT])
async def get_content(message: types.Message, state: FSMContext):
    await state.update_data(id_chat=message.chat.id)
    pict_dir = f"{pict_dir_name}{message.chat.id}"

    if message['photo']:
        content_picture = f"{pict_dir}/{message.photo[-1].file_id}.jpg"
        await message.photo[-1].download(destination_file=f"{content_picture}")

    elif message['document']:
        content_picture = f"{pict_dir}/{message.document.file_name}"
        await message.document.download(destination_file=content_picture)

    else:
        await state.finish()
        await message.answer('Wrong format! \nTry again.')
        return

    await state.update_data(content=content_picture)
    await message.answer('Отправьте мне изображение с которого хотите скопировать стиль')
    await Transform.style.set()


@dp.message_handler(state=Transform.style, content_types=[types.ContentType.PHOTO, types.ContentType.DOCUMENT])
async def get_style(message: types.Message, state: FSMContext):
    pict_dir = f"{pict_dir_name}{message.chat.id}"
    if message['photo']:
        style_picture = f"{pict_dir}/{message.photo[-1].file_id}.jpg"
        await message.photo[-1].download(destination_file=f"{style_picture}")

    elif message['document']:
        style_picture = f"{pict_dir}/{message.document.file_name}"
        await message.document.download(destination_file=style_picture)

    else:
        await state.finish()
        await message.answer('Wrong format! \nTry again /start')
        return
    await state.update_data(style=style_picture)

    await message.answer('Выберите глубину копирования стиля. Чем глубже копирование стиля, тем дольше будет идти обработка', reply_markup=inline_keyboard_confirm)


@dp.callback_query_handler(text=['50epochs', '300epochs', '600epochs'], state='*')
async def agree(call: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    content = data.get('content')
    style = data.get('style')
    id_chat = data.get('id_chat')

    if call.data == '50epochs':
        epochs = 50
        sec = 15
        deep = 'не глубокое'
    elif call.data == '300epochs':
        epochs = 300
        sec = 40
        deep = 'среднее'
    elif call.data == '600epochs':
        epochs = 600
        sec = 60
        deep = 'глубокое'
    else:
        await state.finish()
        # delete pictures
        for file in [content, style]:
            await delete_pict(f"{file}")

        await call.message.answer(f"/start for another picture")
        return

    await state.finish()
    await call.message.answer(f"Вы выбрали: {deep} копирование стиля. Это займет около {sec} секунд...")
    """to neural transfer"""

    out = await run_style_transfer(content, style, num_steps=epochs)

    pict = types.InputFile(path_or_bytesio=f"{out}")
    await call.message.answer(f"Готово")
    await call.message.answer_photo(photo=pict)
    with open(out, 'rb') as picture:
        await call.message.answer_document(document=picture)

    # delete pictures from hdd
    for file in [content, style, out]:
        await delete_pict(f"{file}")
    await delete_pict(f"{pict_dir_name}{id_chat}", directory=True)
    await call.message.answer(text='Выберите действие: ', reply_markup=start_buttons)


##################
"""GAN part"""
##################


@dp.message_handler(text='StyleGAN')
async def ntn(message: types.Message):

    await message.answer(text='Выбран режим: StyleGAN', reply_markup=stop_button)
    await message.answer('Отправьте мне изображение, которое хотите обработать')
    await Gan.content_gan.set()


@dp.message_handler(state=Gan.content_gan, content_types=[types.ContentType.PHOTO, types.ContentType.DOCUMENT])
async def get_content(message: types.Message, state: FSMContext):
    await state.update_data(id_chat_gan=message.chat.id)
    pict_dir = f"{pict_dir_name}{message.chat.id}"
    if message['photo']:
        content_picture = f"{pict_dir}/{message.photo[-1].file_id}.jpg"
        await message.photo[-1].download(destination_file=f"{content_picture}")

        get_img_gan(content_picture, title=content_picture)

        await state.update_data(content_gan=f"{message.photo[-1].file_id}.jpg")

    elif message['document']:
        await message.document.download(destination_file=message.document.file_name)
        content_picture = f"{pict_dir}/{message.document.file_name}"
        await state.update_data(content=content_picture)

    else:
        await state.finish()
        await message.answer('Wrong format! \nTry again.')
        return

    await Gan.style_gan.set()
    await message.answer('Выбери стиль', reply_markup=inline_keyboard_gan)


@dp.callback_query_handler(text=['style_monet_pretrained', 'style_vangogh_pretrained', 'style_cezanne_pretrained', 'style_ukiyoe_pretrained',
                                 'summer2winter_yosemite_pretrained', 'winter2summer_yosemite_pretrained'],
                           state=Gan.style_gan)
async def style_gan(call: types.CallbackQuery, state: FSMContext):
    await state.update_data(style_gan=call.data)

    data = await state.get_data()
    content = data.get('content_gan')
    style = data.get('style_gan')
    id_chat = data.get('id_chat_gan')

    await call.message.answer(f"Применяю стиль к исходному изображению...")

    # Prepare request for StyleGAN
    path = f'python pytorch-CycleGAN-and-pix2pix/test.py ' \
           f'--dataroot ./data{id_chat} ' \
           f'--name pretrained_models/{style} ' \
           f'--model test --no_dropout ' \
           f'--gpu_ids {gpu()} ' \
           f'--results_dir ./result/ ' \
           f'--load_size 512 ' \
           f'--display_winsize 512 ' \
           f'--crop_size 512'
    os.system(path) # send params for StyleGAN and start
    created_path = f"./result/pretrained_models/{style}/test_latest/images/{content[:-4]}_fake.png"
    pict = types.InputFile(path_or_bytesio=created_path)
    await call.message.answer(f"Готово")
    await call.message.answer_photo(photo=pict)
    with open(created_path, 'rb') as picture:
        await call.message.answer_document(document=picture)

    await state.finish()

    await delete_pict(f"{pict_dir_name}{id_chat}/{content}")
    await delete_pict(f"{created_path}")
    await delete_pict(f"{created_path[:-8]}real.png")
    await delete_pict(f"{pict_dir_name}{id_chat}", directory=True)

    await call.message.answer(text='Выберите действие: ', reply_markup=start_buttons)


@dp.callback_query_handler(text='cancel_gan_code', state='*')
async def cancel(call: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    content = data.get('content_gan')
    await state.finish()
    await delete_pict(f"{content}")
    await delete_pict(f"data{call.message.chat.id}", directory=True)
    await call.message.answer(text='Выберите действие: ', reply_markup=start_buttons)


@dp.callback_query_handler(text='cancel_code', state='*')
async def cancel(call: types.CallbackQuery, state: FSMContext):
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
    await state.finish()
    await delete_pict(f"data{message.chat.id}", directory=True)
    await message.answer('Выберите действие: ', reply_markup=start_buttons)


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.answer(text=f'Привет, {message.chat.username}!\n'
                              f'Бот разработан студенотом Школы глубокого обучения МФТИ в рамках дипломного проекта "Телеграм-боты"'
                              'Бот осуществляет стилизацию изображений двумя способами:\n'
                              '1 - Neural Transfer - перенос стиля с одного изображения на другое\n'
                              '2 - StyleGAN - применение заготовленных стилей к изображению\n'
                              '/help - подробное описание работы бота')
    await message.answer(text='Выберите действие: ', reply_markup=start_buttons)


@dp.message_handler(Text(equals=['help', '/help'], ignore_case=True))
async def send_welcome(message: types.Message):

    await message.answer(text='Бот поддерживает два режима работы:\n'
                              '1 - Neural Transfer - перенос стиля с одного изображения на другое\n'
                              '2 - StyleGAN - применение заготовленных стилей к изображению')
    await message.answer(text=f"Для режима Neural Transfer в качестве стиля рекомендуется использовать однородное изображение.\n"
                              f"Также доступен выбор глубины переноса стиля:\n"
                              f"- 'не глубоко' - перенос стиля крайне поверхностный\n"
                              f"- 'средне' - рекомендуемая глубина переноса стиля, стиль просматривается на исходном изображении\n"
                              f"- 'глубоко' - стиль преобладает над исходным изображением\n"
                              f"В зависимости от выбранной глубины перенос занимает от 15 секунд до двух минут.")
    await message.answer(text=f"Для режима StyleGAN доступны следующие стили:\n"
                              f"- Monet\n"
                              f"- Vangogh\n"
                              f"- Cezanne\n"
                              f"- ukiyoe (картины (образы) изменчивого мира) - направление в изобразительном искусстве Японии\n"
                              f"- Лето -> Зима\n"
                              f"- Зима -> Лето\n"
                              f"Обработка изображения занимает около 10 секунд")
    await message.answer(text='Выберите действие: ', reply_markup=start_buttons)


@dp.message_handler()
async def echo(message: types.Message):

    await message.answer(f"/start для начала работы")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
