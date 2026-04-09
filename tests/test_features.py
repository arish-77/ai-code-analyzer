from analyzer.features import extract_features

sample = """
def greet(name):
    return f"Hello, {name}"

def test():
    for i in range(10):
        if i > 5:
            print(i)
"""

print(extract_features(sample))