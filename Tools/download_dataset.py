import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

dataset_urls = [
    "https://zenodo.org/records/3723295/files/subset0.zip?download=1",
    "https://zenodo.org/records/3723295/files/subset1.zip?download=1",
    "https://zenodo.org/records/3723295/files/subset2.zip?download=1",
    "https://zenodo.org/records/3723295/files/subset3.zip?download=1",
    "https://zenodo.org/records/3723295/files/subset4.zip?download=1",
    "https://zenodo.org/records/3723295/files/subset5.zip?download=1",
    "https://zenodo.org/records/3723295/files/subset6.zip?download=1",
    "https://zenodo.org/records/4121926/files/subset7.zip?download=1",
    "https://zenodo.org/records/4121926/files/subset8.zip?download=1",
    "https://zenodo.org/records/4121926/files/subset9.zip?download=1",
    "https://zenodo.org/records/3723295/files/candidates.csv?download=1",
    "https://zenodo.org/records/3723295/files/sampleSubmission.csv?download=1",
    "https://zenodo.org/records/3723295/files/annotations.csv?download=1",
    "https://zenodo.org/records/3723295/files/candidates_V2.zip?download=1",
    "https://zenodo.org/records/3723295/files/seg-lungs-LUNA16.zip?download=1",
    "https://zenodo.org/records/3723295/files/evaluationScript.zip?download=1",
]

def download_file(url):
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'Dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Downloading {url}")
    filename = os.path.basename(url).split("?")[0]
    filepath = os.path.join(dataset_dir, filename)
    urllib.request.urlretrieve(url, filepath)
    print(f"Finished downloading {url} to {filepath}")

with ThreadPoolExecutor(max_workers=32) as executor:
    executor.map(download_file, dataset_urls)