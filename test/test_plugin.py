import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from plugin import register_converters
from markitdown import MarkItDown

md = MarkItDown()
print(len(md._converters))
register_converters(md)
print(len(md._converters))
for conv in md._converters:
    print(type(conv), getattr(conv, "__class__", None), getattr(conv, "__name__", None))