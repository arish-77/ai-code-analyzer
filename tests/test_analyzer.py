from analyzer.parser import analyze

sample = """
def process(data):
    unused = 42          # unused variable

    result = []
    for item in data:
        if item > 0:
            if item > 10:
                if item > 100:
                    if item > 1000:   # depth = 4 → flagged
                        result.append(item)
    return result

def bloated():           # long function → flagged if > 20 lines
    a = 1
    b = 2
    c = 3
    d = 4
    e = 5
    f = 6
    g = 7
    h = 8
    i = 9
    j = 10
    k = 11
    l = 12
    m = 13
    n = 14
    o = 15
    p = 16
    q = 17
    r = 18
    s = 19
    t = 20
    return a
"""

import json
print(json.dumps(analyze(sample), indent=2))