#!/usr/bin/env python3
#
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transmorph",  # Replace with your own username
    version="0.0.7",
    author="Aziz Fouché",
    author_email="aziz.fouche@curie.fr",
    description="Optimal transport-based tools for data integration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Risitop/transmorph",
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'pot',
        'scipy',
        'osqp',
        'sklearn'
    ],
    python_requires=">=3.6",
)
