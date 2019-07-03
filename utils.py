import re


def extract_number(identifier, string):

    pattern = re.compile(r'\d+')
    r = pattern.findall(string)

    start = string.find(identifier)
    result = ''
    while string[start] != '-' and start < len(string)-1:
        start += 1
        result += string[start]

    if result[-1] == '-':
        return result[:-1]
    else:
        return result
    # if identifier == 'l':
    #
    #     return int(r[0])
    # elif identifier == 'p':
    #     return int(r[1])
    # elif identifier == 's':
    #     return int(r[2])
    # else:
    #     return int(r[3])


# print(extract_number('p', 'l0-p00-s0'))

