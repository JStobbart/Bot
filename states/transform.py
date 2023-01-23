from aiogram.dispatcher.filters.state import StatesGroup, State

"""
классы Finite state machine
Для двух нейросетей

"""


class Transform(StatesGroup):
    id_chat = State()
    content = State()
    style = State()
    transformation = State()


class Gan(StatesGroup):
    id_chat_gan = State()
    content_gan = State()
    style_gan = State()
    transformation_gan = State()
