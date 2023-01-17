from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

inline_keyboard_gan = InlineKeyboardMarkup(row_width=2, inline_keyboard=[
    [
        InlineKeyboardButton(text='Monet', callback_data='style_monet_pretrained'),
        InlineKeyboardButton(text='Vangogh', callback_data='style_vangogh_pretrained')
    ],
    [
        InlineKeyboardButton(text='Cezanne', callback_data='style_cezanne_pretrained'),
        InlineKeyboardButton(text='Ukiyoe', callback_data='style_ukiyoe_pretrained')
    ],
    [
        InlineKeyboardButton(text='cancel', callback_data='cancel_gan_code')
    ]
])

inline_keyboard_confirm = InlineKeyboardMarkup(row_width=2, inline_keyboard=[
    [
        InlineKeyboardButton(text='50', callback_data='50epochs'),
        InlineKeyboardButton(text='300', callback_data='300epochs')
    ],
    [
        InlineKeyboardButton(text='500', callback_data='500epochs'),
        InlineKeyboardButton(text='cancel', callback_data='cancel_code')
    ]
])
