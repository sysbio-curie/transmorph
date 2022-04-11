#!/usr/bin/env python3

from transmorph.datasets import load_bank, remove_bank


def test_download_zhou10x():
    remove_bank("zhou_10x")
    _ = load_bank("zhou_10x")


def test_download_chen10x():
    remove_bank("chen_10x")
    _ = load_bank("chen_10x")


if __name__ == "__main__":
    test_download_chen10x()
