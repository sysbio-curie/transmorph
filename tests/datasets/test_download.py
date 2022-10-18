#!/usr/bin/env python3

from transmorph.datasets import load_bank, remove_bank

ENABLE = False


def test_download_chen10x():
    if not ENABLE:
        return
    remove_bank("chen_10x")
    _ = load_bank("chen_10x")


def test_download_pal10x():
    if not ENABLE:
        return
    remove_bank("pal_10x")
    _ = load_bank("pal_10x")


def test_download_travaglini10x():
    if not ENABLE:
        return
    remove_bank("travaglini_10x")
    _ = load_bank("travaglini_10x")


def test_download_zhou10x():
    if not ENABLE:
        return
    remove_bank("zhou_10x")
    _ = load_bank("zhou_10x")


if __name__ == "__main__":
    test_download_pal10x()
