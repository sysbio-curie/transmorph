#!/usr/bin/env python3

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transmorph",
    version="0.2.0",
    author="Aziz Fouché, Loïc Chadoutaud (Institut Curie, Paris)",
    author_email="aziz.fouche@curie.fr",
    description="Optimal transport-based data integration.",
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
        "anndata>=0.8.0",
        "igraph",
        "leidenalg",
        "numpy<1.21,>=1.17",
        "pre-commit",
        "pot",
        "pymde",
        "pynndescent",
        "scanpy",
        "scikit-learn",
        "scipy",
        "stabilized-ica",
        "umap-learn",
    ],
    python_requires=">=3.9,<3.10",
)
