#!/usr/bin/env python3

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transmorph",
    version="0.2.5",
    author="Aziz Fouché, Loïc Chadoutaud, Andrei Zinovyev (Institut Curie, Paris)",
    author_email="aziz.fouche@curie.fr",
    description="A unifying data integration framework.",
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
        "cython",
        "igraph",
        "leidenalg",
        "numpy>=1.22",
        "pre-commit",
        "pot",
        "pymde",
        "pynndescent",
        "scanpy",
        "sccover",
        "scikit-learn",
        "scipy",
        "stabilized-ica>=2.0",
        "umap-learn",
    ],
    python_requires=">=3.9",
)
