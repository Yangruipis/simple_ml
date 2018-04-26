# -*- coding:utf-8 -*-


def chinese(data):
    """
    获取字符串长度，中文算两个字符串，其他算一个
    :param data:   str
    :return:       int
    """
    count = 0
    for s in data:
        if ord(s) > 127:
            count += 2
        else:
            count += 1
    return count


def log_print(s):
    """
    日志打印，根据中文字符串长度自动补全"="，好看一点（强迫症）
    :param s:    str
    :return:     void
    """
    l = chinese(s)
    num = (70 - l) // 2
    print("="*num + " " + s + " " + "="*num)
