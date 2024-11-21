import discord
from discord.ext import commands

import time
import timeago
import datetime
from datetime import datetime

from collections import namedtuple

import os
import psutil
try:
    from utils import repo
    from models.trainer import Trainer
except ModuleNotFoundError:
    import sys
    sys.path.append(sys.path[0] + '/..')
    from utils import repo
    from models.trainer import Trainer



class Information(commands.Cog):
    """ Prints out useful information about the bot """
    def __init__(self, bot):
        self.bot = bot
        self.process = psutil.Process(os.getpid())
        if not hasattr(self.bot, 'uptime'):
            self.bot.uptime = datetime.datetime.now()
            
    def _timeago(time_interval):
        """Time since time interval
        
        Args:
            time_interval (datetime): time interval

        Returns:
            timeago: time between time interval
        """
        return timeago.format(time_interval)

    @commands.command()
    async def ping(self, ctx):
        """Measures ping
        
        Args:
            ctx (discord.context): context
        """
        before = time.monotonic()
        message = await ctx.send('Pong!')
        ping = (time.monotonic() - before)
        await message.edit(content=f'Pong! | `{int(ping) * 1000} ms`')

    @commands.command(aliases=['botinfo', 'botstats', 'botstatus', 'about'])
    async def aboutbot(self, ctx):
        """ About the bot 
        
        Args:
            ctx (discord.context): context
        """
        ram_usage = self.process.memory_full_info().rss / 1024**2
        avg_members = round(len(self.bot.users) / len(self.bot.guilds))

        embed = discord.Embed(colour=ctx.me.top_role.colour)
        embed.set_thumbnail(url=ctx.bot.user.avatar_url)
        embed.add_field(name='About', value='I am a convolutional neural network that has been trained specifically to detect images of foxes.')
        embed.add_field(name='Training Instances:', name=len(os.listdir('fox-data/train')))
        embed.add_field(name='Parameters:', name='{:,}'.format(Trainer().count_parameters()))
        embed.add_field(name='Neurons:', value='{:,}'.format(Trainer().count_neurons()))
        embed.add_field(name="Last boot", value=self._timeago(datetime.now() - self.bot.uptime), inline=True)
        embed.add_field(name="Developer: Julianne Cai", inline=True)
        embed.add_field(name="Powered by:", value="discord.py, PyTorch", inline=True)
        embed.add_field(name="Servers", value=f"{len(ctx.bot.guilds)} ( avg: {avg_members} users/server )", inline=True)
        # embed.add_field(name="Commands loaded", value=str(len([x.name for x in self.bot.commands])), inline=True)
        embed.add_field(name="RAM", value=f"{ram_usage:.2f} MB", inline=True)

        await ctx.send(content=f"ℹ About **{ctx.bot.user}** | **{repo.VERSION_DATA['Version']}**", embed=embed)
        
    def setup(bot):
        bot.add_cog(Information(bot))
        