from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

start_buttons = ReplyKeyboardMarkup(keyboard=[
    [
        KeyboardButton(text='Neural transfer')

    ],
    [
        KeyboardButton(text='StyleGAN')
    ],
    [
        KeyboardButton(text='help')
    ]
], resize_keyboard=True, row_width=3)

stop_button = ReplyKeyboardMarkup(keyboard=[
    [
        KeyboardButton(text='Stop')

    ]
], resize_keyboard=True, row_width=3)