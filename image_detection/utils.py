# coding: utf-8
#!/usr/bin/env python
from termcolor import colored

warning_color = lambda x: colored(x, 'red', attrs=['reverse', 'blink'])
hint_color = lambda x: colored(x, 'cyan')


# TIME TEST
def timecost_of(k, func, *args, **kwargs):
    '''Test function run time, and return evalution of function.'''
    from time import time
    start = time()
    for i in range(k):
        res = func(*args)
    stop = time()
    print("Cost: {:.04f} sec".format((stop - start) / k))
    return (res, (stop - start)/k)


def input_with_y_or_n(hint, default):
    while True:
        x = input(hint_color(hint)) or default
        if x.lower() == 'y':
            return True
        elif x.lower() == 'n':
            return False


def input_until_input(msg):
    while True:
        x = input(msg)
        if x:
            return x
