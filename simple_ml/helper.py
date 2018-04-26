# -*- coding:utf-8 -*-


def chinese(data):
    count = 0
    for s in data:
        if ord(s) > 127:
            count += 2
        else:
            count += 1
    return count


def log_print(s):
    l = chinese(s)
    num = (70 - l) // 2
    print("="*num + " " + s + " " + "="*num)
