from random import choice


__all__ = ['red', 'blue', 'cyan', 'green', 'bold']


RED = "\033[1;31m"
BLUE = "\033[1;34m"
CYAN = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD = "\033[;1m"
REVERSE = "\033[;7m"


def red(msg):
    return RED + msg + RESET


def blue(msg):
    return BLUE + msg + RESET


def cyan(msg):
    return CYAN + msg + RESET


def green(msg):
    return GREEN + msg + RESET


def bold(msg):
    return BOLD + msg + RESET


def reverse(msg):
    return REVERSE + msg + RESET


def random(msg):
    color_fn = choice([red, blue, cyan, green, bold, reverse])
    return color_fn(msg)
