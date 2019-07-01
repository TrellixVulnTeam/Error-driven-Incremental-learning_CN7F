import re


def extract_number(identifier, string):

    pattern = re.compile(r'\d+')
    r = pattern.findall(string)

    if identifier == 'l':
        return int(r[0])
    elif identifier == 'p':
        return int(r[1])
    elif identifier == 's':
        return int(r[2])
    else:
        return int(r[3])

extract_number('p', 'l0-p10-s3-12')
