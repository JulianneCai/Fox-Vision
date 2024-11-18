import discord
from discord.ext import commands

import time
import timeago
import datetime
from datetime import datetime

from collections import namedtuple

import os
import psutil

from utils import repo


class Information(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.process = psutil.Process(os.getpid())
        if not hasattr(self.bot, 'uptime'):
            self.bot.uptime = datetime.datetime.now()
            
    def _timeago(time_interval):
        """
        Time since time interval
        Args:
            time_interval (datetime): time interval

        Returns:
            timeago: time between time interval
        """
        return timeago.format(time_interval)

    @commands.command()
    async def ping(self, ctx):
        """Measures ping"""
        before = time.monotonic()
        message = await ctx.send('Pong!')
        ping = (time.monotonic() - before)
        await message.edit(content=f'Pong! | `{int(ping) * 1000} ms`')

    @commands.command(aliases=['botinfo', 'botstats', 'botstatus', 'about'])
    async def aboutbot(self, ctx):
        """ About the bot """
        ram_usage = self.process.memory_full_info().rss / 1024**2
        avg_members = round(len(self.bot.users) / len(self.bot.guilds))

        embed = discord.Embed(colour=ctx.me.top_role.colour)
        embed.set_thumbnail(url=ctx.bot.user.avatar_url)
        embed.add_field(name='About', value='I am a convolutional neural network that has been trained specifically to detect images of foxes.')
        embed.add_field(name='Training Instances:', name=len(os.listdir('fox-data/train')))
        embed.add_field(name="Last boot", value=self._timeago(datetime.now() - self.bot.uptime), inline=True)
        embed.add_field(name="Developer: Julianne Cai", inline=True)
        embed.add_field(name="Library", value="discord.py", inline=True)
        embed.add_field(name="Servers", value=f"{len(ctx.bot.guilds)} ( avg: {avg_members} users/server )", inline=True)
        embed.add_field(name="Commands loaded", value=str(len([x.name for x in self.bot.commands])), inline=True)
        embed.add_field(name="RAM", value=f"{ram_usage:.2f} MB", inline=True)

        await ctx.send(content=f"ℹ About **{ctx.bot.user}** | **{repo.VERSION_DATA['Version']}**", embed=embed)
        