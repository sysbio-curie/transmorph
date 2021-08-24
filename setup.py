#!/usr/bin/env python3
#
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transmorph",
    version="0.1.0",
    author="Aziz Fouché (Institut Curie, Paris)",
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
    include_package_data=True,
    install_requires=[
        'numpy<1.21,>=1.17',
        'cython',
        'pot',
        'scipy',
        'osqp',
        'sklearn',
        'numba',
    ],
    python_requires=">=3.7",
)
