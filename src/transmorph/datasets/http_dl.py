import base64
import json
import os
import time
import urllib.request
import zipfile

from os import path
from posixpath import dirname
from socket import error as SocketError

CALLBACK = 30
DATASETS_JSON = "datasets.json"


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


def download_dataset(dataset_name: str) -> None:
    # Downloads a dataset from onedrive
    # using data from datasets.json
    # TODO print -> log

    # Detecting package structure
    datasets_root = dirname(__file__) + "/%s"

    # Ensuring dataset exists
    f = open(datasets_root % DATASETS_JSON, "r")
    all_datasets = json.load(f)
    dataset = None
    for ds in all_datasets:
        if ds["name"] == dataset_name:
            dataset = ds
            break
    assert dataset is not None, f"http_dl > Unknown dataset: {dataset_name}."
    print(f"http_dl > Loading dataset {dataset_name}...")
    print(f"http_dl > Reference: {dataset['reference']}")
    print(f"http_dl > Type: {dataset['type']}")
    print(f"http_dl > Number of files: {dataset['n_files']}")

    # Preparing dir structure
    if not path.exists(datasets_root % "data/"):
        print("http_dl > data/ not found, creating it.")
        os.mkdir(datasets_root % "data/")
    data_path = datasets_root % ("data/%s" % dataset_name) + "/"
    if not path.exists(data_path):
        print(f"http_dl > data/{dataset_name}/ not found, creating it.")
        os.mkdir(data_path)

    # Loading dataset TODO 100% zip files
    downloads = 0
    is_zip = "zip_link" in dataset
    if not is_zip:
        for dfile in dataset["files"]:
            if path.exists(data_path + dfile["name"]):
                continue
            print(f"http_dl > Downloading file {dfile['name']}...")
            dl_url = url_to_dllink(dfile["url"])
            urllib.request.urlretrieve(dl_url, data_path + dfile["name"])
            downloads += 1
    else:
        dl_url = url_to_dllink(dataset["zip_link"])
        print(f"http_dl > Downloading file {dataset['zip_name']}...")
        try:
            urllib.request.urlretrieve(dl_url, data_path + dataset["zip_name"])
        except SocketError as e:
            print(f"http_dl > # ERROR # {e}")
            print(f"http_dl > Retrying in {CALLBACK} seconds.")
            time.sleep(CALLBACK)
            download_dataset(dataset_name)
            return
        print(f"http_dl > Unzipping file {dataset['zip_name']}...")
        with zipfile.ZipFile(data_path + dataset["zip_name"], "r") as fzip:
            fzip.extractall(data_path)
        os.remove(data_path + dataset["zip_name"])

    f.close()
    if downloads > 0:
        print(f"http_dl > Dataset {dataset_name} successfully downloaded.")
    else:
        print("http_dl > Complete dataset found, nothing to do.")


if __name__ == "__main__":
    download_dataset("travaglini_10x")
