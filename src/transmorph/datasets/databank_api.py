import base64
import hashlib
import json
import os
import time
import urllib.request
import zipfile

from os import path
from posixpath import dirname
from socket import error as SocketError
from typing import Dict, Optional
from warnings import warn

CALLBACK = 30
DATASETS_JSON = "datasets.json"
DATA_PATH = f"{dirname(__file__)}/data/"


def check_md5_file(file_path: str, target_hash: str) -> bool:
    # Returns a md5 hash to quality check
    file_hash = hashlib.md5(open(file_path, "rb").read()).hexdigest()
    return file_hash == target_hash


def remove_dataset(dataset_name: str) -> None:
    # Removes all files from data/$dataset_name
    print(f"databank_api > Removing dataset {dataset_name}...")
    if not path.exists(f"{DATA_PATH}{dataset_name}"):
        print("databank_api > Nothing to do.")
        return
    for fname in os.listdir(f"{DATA_PATH}{dataset_name}/"):
        os.remove(f"{DATA_PATH}{dataset_name}/{fname}")
    os.rmdir(f"{DATA_PATH}{dataset_name}")


def unzip_file(zip_path: str, dataset_name: str) -> None:
    # Unzips $zip_path at data/$dataset_name/
    print(f"databank_api > Unzipping file {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as fzip:
        fzip.extractall(f"{DATA_PATH}{dataset_name}/")
    os.remove(zip_path)


def check_files(dataset_name: str) -> bool:
    # Check if all files from a dataset are present
    print(f"databank_api > Checking files from {dataset_name}...")
    with open(f"{dirname(__file__)}/{DATASETS_JSON}", "r") as f:
        all_datasets = json.load(f)
        dataset: Optional[Dict] = None
        for ds in all_datasets:
            if ds["name"] == dataset_name:
                dataset = ds
                break
        if dataset is None:
            warn(f"Dataset {dataset_name} not found in database.")
            return False
        for fname in dataset["files"]:
            if not path.exists(f"{dirname(__file__)}/data/{dataset_name}/{fname}"):
                return False
    return True


# From https://towardsdatascience.com/
#      how-to-get-onedrive-direct-download-link-ecb52a62fee4
def url_to_dllink(onedrive_link: str) -> str:
    # Converts a "share" link to a direct download link
    data_bytes64 = base64.b64encode(bytes(onedrive_link, "utf-8"))
    data_bytes64_String = (
        data_bytes64.decode("utf-8").replace("/", "_").replace("+", "-").rstrip("=")
    )
    resultUrl = (
        f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    )
    return resultUrl


def download_dataset(dataset_name: str) -> str:
    # Downloads a dataset from onedrive
    # using data from datasets.json
    # TODO print -> log

    datasets_root = f"{dirname(__file__)}/"

    # Ensuring dataset exists
    f = open(f"{datasets_root}{DATASETS_JSON}", "r")
    all_datasets = json.load(f)
    dataset = None
    for ds in all_datasets:
        if ds["name"] == dataset_name:
            dataset = ds
            break
    assert dataset is not None, f"databank_api > Unknown dataset: {dataset_name}."
    print(f"databank_api > Loading dataset {dataset_name}...")
    print(f"databank_api > Reference: {dataset['reference']}")
    print(f"databank_api > Type: {dataset['type']}")
    print(f"databank_api > Number of files: {dataset['n_files']}")

    # Preparing dir structure
    if not path.exists(f"{datasets_root}data/"):
        print("databank_api > data/ not found, creating it.")
        os.mkdir(f"{datasets_root}data/")
    data_path = f"{datasets_root}data/{dataset_name}/"
    if not path.exists(data_path):
        print(f"databank_api > data/{dataset_name}/ not found, creating it.")
        os.mkdir(data_path)

    # Loading dataset
    dl_url = url_to_dllink(dataset["zip_link"])
    print(f"databank_api > Downloading file {dataset['zip_name']}...")
    zip_path = f"{data_path}{dataset['zip_name']}"
    try:
        urllib.request.urlretrieve(dl_url, zip_path)
    except SocketError as e:
        f.close()
        print(f"databank_api > # ERROR # {e}")
        print(f"databank_api > Retrying in {CALLBACK} seconds.")
        time.sleep(CALLBACK)
        return download_dataset(dataset_name)

    print("databank_api > Checking md5 sums...")
    if not check_md5_file(zip_path, dataset["md5_hash"]):
        f.close()
        os.remove(zip_path)
        print("databank_api > # ERROR # Errors detected in the file.")
        print(f"databank_api > Retrying in {CALLBACK} seconds.")
        time.sleep(CALLBACK)
        return download_dataset(dataset_name)

    f.close()
    print(f"databank_api > Dataset {dataset_name} successfully downloaded.")
    return zip_path