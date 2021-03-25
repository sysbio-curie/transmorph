#!/usr/bin/env python3
#
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="woti",  # Replace with your own username
    version="0.0.1",
    author="Aziz Fouché",
    author_email="aziz.focuche@curie.fr",
    description="Unbalanced optimal transport dataset integration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pot',
        'scipy',
        'osqp',
        'awkde @ git+https://github.com/mennthor/awkde',
    ],
    python_requires=">=3.6",
)
