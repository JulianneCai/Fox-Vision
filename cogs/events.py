import discord
import re
import datetime

from discord.ext import commands as bot_commands
from discord.ext.commands import errors


class Events(bot_commands.Cog):
    """ Handles reactions to message events """
    def __init__(self, bot):
        self.bot = bot
        
    async def on_ready(self):
        """ Bot behaviour when it first starts up """
        if not hasattr(self.bot, 'uptime'):
            self.uptime = datetime.datetime.now()
            
        activity_message = discord.Game('Learning about foxes. Try fox.help.')
        await self.bot.change_presence(status=discord.Status.online, activity=activity_message)
    
    async def on_command_error(self, ctx, error: errors):
        """Bot behaviour when encountering error running command

        Args:
            ctx (discord.ext.commands.Context): context
            error (errors): error type

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    
    async def on_message(self, message):
        """Bot beh

        Args:
            message (_type_): _description_
        """
        if re.search(r'\b(fox)\b', message.content, re.IGNORECASE):
            await message.add_reaction('🦊')
            
    def setup(bot):
        bot.add_cog(Events(bot))
            