import discord


def can_send(ctx) -> bool:
    """
    Checks whether the bot can send messages in a channel.

    Params:
        ctx (discord.commands.Context): context
        
    Returns:
        bool: True if bot can send message, False otherwise
    """
    return isinstance(ctx.channel, discord.DMChannel) or ctx.channel.permissions_for(ctx.guild.me).send_messages


def can_embed(ctx) -> bool:
    """Checks whether bot can send embedded images into current channel
    
    Args:
        ctx (discord.commands.Context): context
        
    Returns:
        bool: True if bot can send message, False otherwise
    """
    return isinstance(ctx.channel, discord.DMChannel) or ctx.channel.permissions_for(ctx.guild.me).embed_links


def can_attach(ctx) -> bool:
    """Checks if bot can attach files in their messages.
    
    Args:
        ctx (discord.commands.Context): context
        
    Returns:
        bool: True if bot can send message, False otherwise
    """
    return isinstance(ctx.channel, discord.DMChannel) or ctx.channel.permissions_for(ctx.guild.me).attach_files


def can_connect_voice(ctx) -> bool:
    """Checks if bot can connect to a voice channel.
    
    Args:
        ctx (discord.commands.Context): context
        
    Returns:
        bool: True if bot can send message, False otherwise
    """
    return isinstance(ctx.channel, discord.DMChannel) or ctx.channel.permissions_for(ctx.guild.me).connect


def is_nsfw(ctx) -> bool:
    """Checks if the channel has NSFW mode enabled
    
    Args:
        ctx (discord.commands.Context): context
        
    Returns:
        bool: True if bot can send message, False otherwise
    """
    return isinstance(ctx.channel, discord.DMChannel) or ctx.channel.is_nsfw()


def can_react(ctx) -> bool:
    """Checks if the bot can react to messages
    
    Args:
        ctx (discord.commands.Context): context
        
    Returns:
        bool: True if bot can send message, False otherwise
    """
    return isinstance(ctx.chaannel, discord.DMChannel) or ctx.channel.permissions_for(ctx.guild.me).add_reactions
