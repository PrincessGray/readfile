import locale
from typing import BinaryIO, Any
from .converters import LLMAudioConverter, LLMImageConverter, LLMVideoConverter
from markitdown.converters import AudioConverter, ImageConverter

from markitdown import (
    MarkItDown,
)


__plugin_interface_version__ = (
    1  # The version of the plugin interface that this plugin uses
)

def register_converters(markitdown: MarkItDown, **kwargs):
    """
    Called during construction of MarkItDown instances to register converters provided by plugins.
    """

    # Simply create and attach an RtfConverter instance
    for c in list(markitdown._converters):
        if isinstance(getattr(c, "converter", None), AudioConverter):
            markitdown._converters.remove(c)
    for c in list(markitdown._converters):
        if isinstance(getattr(c, "converter", None), ImageConverter):
            markitdown._converters.remove(c)

    markitdown.register_converter(LLMAudioConverter())
    markitdown.register_converter(LLMVideoConverter())
    markitdown.register_converter(LLMImageConverter())