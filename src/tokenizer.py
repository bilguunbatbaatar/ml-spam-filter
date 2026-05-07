import re


def tokenize(text):
    text = text.lower()
    return re.findall(r"[a-zA-Z0-9]+", text)