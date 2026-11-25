"""Download and extract the raw 9-channel UCI HAR Dataset into data/."""
import os
import urllib.request
import zipfile
from pathlib import Path

UCI_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"


def main():
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    if any(data_dir.rglob("Inertial Signals")):
        print("UCI HAR raw inertial signals are already extracted in data/.")
        return

    outer_zip = data_dir / "uci_har_download.zip"
    print(f"Downloading UCI HAR archive from:\n  {UCI_URL}")
    urllib.request.urlretrieve(UCI_URL, outer_zip)

    print("Extracting outer archive...")
    with zipfile.ZipFile(outer_zip, "r") as zf:
        zf.extractall(data_dir)

    for inner_zip in data_dir.rglob("UCI HAR Dataset.zip"):
        print(f"Extracting nested archive: {inner_zip}...")
        with zipfile.ZipFile(inner_zip, "r") as zf_in:
            zf_in.extractall(data_dir)

    os.remove(outer_zip)
    print("Done! Raw inertial signal files are ready under data/UCI HAR Dataset/.")


if __name__ == "__main__":
    main()
