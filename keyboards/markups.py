from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

start_buttons = ReplyKeyboardMarkup(keyboard=[
    [
        KeyboardButton(text='Neural transfer net')

    ],
    [
        KeyboardButton(text='Style GAN')
    ],
    [
        KeyboardButton(text='Cancel')
    ]
], resize_keyboard=True, row_width=3)
