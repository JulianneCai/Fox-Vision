from discord.ext import commands as bot_commands


VERSION_DATA = {
    'Colour': 'Sinopia',
    'Version': 1,
    'Build': 0,
    'ColourHex': 0xcb410b
}

ONLINE_STATUS = 'Online'

def get_prefixes(bot, message):
    PREFIXES = ['fox']
    return bot_commands.when_mentioned_or(*PREFIXES)(bot, message)
