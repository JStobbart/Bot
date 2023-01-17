import os
import gc
import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text

from keyboards import inline_keyboard_confirm, inline_keyboard_gan, start_buttons

from states import Transform, Gan

from net import run_transfer, get_cnn, get_device, delete_pict, get_img_gan, gpu


# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=os.environ.get('API_TOKEN'))
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
pict_dir_name = 'data'


async def run_style_transfer(content, style, num_steps):
    device = get_device()
    cnn = get_cnn(device)
    out = run_transfer(cnn=cnn, content=content, style=style, num_steps=num_steps,
                       title=f'{content[:-4]}_styled.jpg')
    del cnn
    gc.collect()
    return out


# @dp.message_handler(commands='start')
# async def transform_picture(message: types.Message):
#     await message.answer('Send me a content picture')
#     await Transform.content.set()


@dp.message_handler(text='Neural transfer net')
async def ntn(message: types.Message):


    await message.answer(text='Neural transfer net', reply_markup=types.ReplyKeyboardRemove())
    await message.answer(f'Send me a content picture')
    await Transform.content.set()


@dp.message_handler(state=Transform.content, content_types=[types.ContentType.PHOTO, types.ContentType.DOCUMENT])
async def get_content(message: types.Message, state: FSMContext):
    await state.update_data(id_chat=message.chat.id)
    pict_dir = f"{pict_dir_name}{message.chat.id}"

    if message['photo']:
        content_picture = f"{pict_dir}/{message.photo[-1].file_id}.jpg"
        await message.photo[-1].download(destination_file=f"{content_picture}")
        print(message.photo)

    elif message['document']:
        await message.document.download(destination_file=message.document.file_name)
        content_picture = message.document.file_name

    else:
        await state.finish()
        await message.answer('Wrong format! \nTry again.')
        return

    await state.update_data(content=content_picture)
    await message.answer('Send me a style picture')
    await Transform.style.set()


@dp.message_handler(state=Transform.style, content_types=[types.ContentType.PHOTO, types.ContentType.DOCUMENT])
async def get_style(message: types.Message, state: FSMContext):
    pict_dir = f"{pict_dir_name}{message.chat.id}"
    if message['photo']:
        style_picture = f"{pict_dir}/{message.photo[-1].file_id}.jpg"
        await message.photo[-1].download(destination_file=f"{style_picture}")
        print(message.photo)

    elif message['document']:
        await message.document.download(destination_file=message.document.file_name)
        style_picture = message.document.file_name

    else:
        await state.finish()
        await message.answer('Wrong format! \nTry again /start')
        return
    await state.update_data(style=style_picture)
    await Transform.transformation.set()
    await message.answer("let's transform...")
    await message.answer('Please, choose the number of epochs', reply_markup=inline_keyboard_confirm)


@dp.callback_query_handler(text=['50epochs', '300epochs', '500epochs'],
                           state=Transform.transformation)
async def agree(call: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    content = data.get('content')
    style = data.get('style')
    id_chat = data.get('id_chat')

    if call.data == '50epochs':
        epochs = 50
    elif call.data == '300epochs':
        epochs = 300
    elif call.data == '500epochs':
        epochs = 500
    else:
        await state.finish()
        # delete pictures
        for file in [content, style]:
            await delete_pict(f"{file}")

        await call.message.answer(f"/start for another picture")
        return

    await state.finish()
    await call.message.answer(f"you chose {epochs} epochs, this may take a while")

    """to neural transfer"""
    out = await run_style_transfer(content, style, num_steps=epochs)

    pict = types.InputFile(path_or_bytesio=f"{out}")
    await dp.bot.send_photo(chat_id=call.message.chat.id, photo=pict)

    # delete pictures
    for file in [content, style, out]:
        await delete_pict(f"{file}")
    await delete_pict(f"{pict_dir_name}{id_chat}", directory=True)
    await call.message.answer(text='Выберите действие: ', reply_markup=start_buttons)


##################
"""GAN part"""
##################


@dp.message_handler(text='Style GAN')
async def ntn(message: types.Message):

    await message.answer(text='Style GAN', reply_markup=types.ReplyKeyboardRemove())
    await message.answer('GAN. Send me a content picture')
    await Gan.content_gan.set()

# @dp.message_handler(commands=['gan'])
# async def send_welcome(message: types.Message):
#     await message.answer('GAN. Send me a content picture')
#     await Gan.content_gan.set()


@dp.message_handler(state=Gan.content_gan, content_types=[types.ContentType.PHOTO, types.ContentType.DOCUMENT])
async def get_content(message: types.Message, state: FSMContext):
    await state.update_data(id_chat_gan=message.chat.id)
    pict_dir = f"{pict_dir_name}{message.chat.id}"
    if message['photo']:
        content_picture = f"{pict_dir}/{message.photo[-1].file_id}.jpg"
        await message.photo[-1].download(destination_file=f"{content_picture}")

        content_picture = get_img_gan(content_picture, title=content_picture)

        print(message.photo)

        await state.update_data(content_gan=f"{message.photo[-1].file_id}.jpg")

    elif message['document']:
        await message.document.download(destination_file=message.document.file_name)
        content_picture = message.document.file_name
        await state.update_data(content=content_picture)

    else:
        await state.finish()
        await message.answer('Wrong format! \nTry again.')
        return

    await Gan.style_gan.set()
    await message.answer('Please, choose the style', reply_markup=inline_keyboard_gan)


@dp.callback_query_handler(text=['style_monet_pretrained', 'style_vangogh_pretrained', 'style_cezanne_pretrained', 'style_ukiyoe_pretrained'],
                           state=Gan.style_gan)
async def style_gan(call: types.CallbackQuery, state: FSMContext):
    await state.update_data(style_gan=call.data)

    data = await state.get_data()
    content = data.get('content_gan')
    style = data.get('style_gan')
    id_chat = data.get('id_chat_gan')


    await call.message.answer(f"you chose {call.data[6:-11]} style, this may take a while. GAN is working... ")

    # path = f'python pytorch-CycleGAN-and-pix2pix/test.py ' \
    #        f'--dataroot /home/stobbart/PycharmProjects/Bot/data{id_chat} ' \
    #        f'--name /home/stobbart/PycharmProjects/Bot/pretrained_models/{style} ' \
    #        f'--model test --no_dropout ' \
    #        f'--gpu_ids {gpu()} ' \
    #        f'--results_dir /home/stobbart/PycharmProjects/Bot/result/ ' \
    #        f'--load_size 512 ' \
    #        f'--display_winsize 512 ' \
    #        f'--crop_size 512'
    path = f'python pytorch-CycleGAN-and-pix2pix/test.py ' \
           f'--dataroot ./data{id_chat} ' \
           f'--name pretrained_models/{style} ' \
           f'--model test --no_dropout ' \
           f'--gpu_ids {gpu()} ' \
           f'--results_dir ./result/ ' \
           f'--load_size 512 ' \
           f'--display_winsize 512 ' \
           f'--crop_size 512'
    os.system(path)
    # created_path = f"/home/stobbart/PycharmProjects/Bot/pretrained_models/{style}/test_latest/images/{content[:-4]}_fake.png"
    created_path = f"./result/pretrained_models/{style}/test_latest/images/{content[:-4]}_fake.png"
    #print(created_path)
    pict = types.InputFile(path_or_bytesio=created_path)
    await dp.bot.send_photo(chat_id=call.message.chat.id, photo=pict)

    await state.finish()

    """to gan transfer"""

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
    await call.message.answer('cancelled')
    await call.message.answer(text='Выберите действие: ', reply_markup=start_buttons)


@dp.callback_query_handler(text='cancel_code', state='*')
async def cancel(call: types.CallbackQuery, state: FSMContext):
    data = await state.get_data()
    content = data.get('content')
    style = data.get('style')
    await state.finish()
    await delete_pict(f"{content}")
    await delete_pict(f"{style}")
    await call.message.answer('cancelled')
    await call.message.answer(text='Выберите действие: ', reply_markup=start_buttons)


@dp.message_handler(Text(equals=['cancel', '/cancel'], ignore_case=True), state='*')
async def cancel(message: types.Message, state: FSMContext):
    await state.finish()
    await message.answer('cancelled by cancel', reply_markup=types.ReplyKeyboardRemove())


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):

    """
    This handler will be called when user sends `/start` or `/help` command
    """
    await message.answer(text='Выберите действие: ', reply_markup=start_buttons)
    # await message.reply("Hi!\nI'm Transfer Style Bor!\n/start for transfer style.")








@dp.message_handler()
async def echo(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)

    await message.answer(f"/start for transfer style")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
