from discord.ext import commands as bot_commands


###  for tuning convolutional neural network ###
DATA_DIR = 'fox-data/train'
BATCH_SIZE = 64
IMG_SIZE = 64

###  Discord bot information ###

VERSION_DATA = {
    'Colour': 'Orange',
    'Version': 1,
    'Build': 0
}

ONLINE_STATUS = 'Online'

def get_prefixes(bot, message):
    PREFIXES = ['fox']
    return bot_commands.when_mentioned_or(*PREFIXES)(bot, message)
