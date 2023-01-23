from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
"""
инлайн клавиатуры

"""
inline_keyboard_gan = InlineKeyboardMarkup(row_width=4, inline_keyboard=[
    [
        InlineKeyboardButton(text='Monet', callback_data='style_monet_pretrained'),
        InlineKeyboardButton(text='Vangogh', callback_data='style_vangogh_pretrained'),
        InlineKeyboardButton(text='Cezanne', callback_data='style_cezanne_pretrained'),
        InlineKeyboardButton(text='Ukiyoe', callback_data='style_ukiyoe_pretrained')
    ],
    [
        InlineKeyboardButton(text='Aivazovsky', callback_data='style_aivazovsky_pretrained')
    ],
    [
        InlineKeyboardButton(text='Лето -> Зима', callback_data='summer2winter_yosemite_pretrained'),
        InlineKeyboardButton(text='Зима -> Лето', callback_data='winter2summer_yosemite_pretrained')

    ],
    [
        InlineKeyboardButton(text='Отмена', callback_data='cancel_gan_code')
    ]
])

inline_keyboard_confirm = InlineKeyboardMarkup(row_width=2, inline_keyboard=[
    [
        InlineKeyboardButton(text='Не глубоко', callback_data='50epochs'),
        InlineKeyboardButton(text='Средне', callback_data='300epochs')
    ],
    [
        InlineKeyboardButton(text='Глубоко', callback_data='600epochs'),
        InlineKeyboardButton(text='Отмена', callback_data='cancel_code')
    ]
])
