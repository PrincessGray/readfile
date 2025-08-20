#!/usr/bin/env python
import os
from setuptools import setup, find_packages

def read_requirements():
    req_path = "requirements.txt"
    if not os.path.exists(req_path):
        return []
    with open(req_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]
setup(
    name="read-file-enhanced",
    version="0.1.0",
    author="princessgray",
    author_email="k1mg0ng2004@gmail.com",
    description="Enhanced file reader with PDF, image, and LLM support.",
    long_description="A tool to read and convert files (PDF, images, etc.) to markdown, with LLM-powered image analysis.",
    long_description_content_type="text/markdown",
    url="https://github.com/princessgray/read-file-enhanced",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=read_requirements(),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "read-file-enhanced=read_file_enhanced.server:main",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)