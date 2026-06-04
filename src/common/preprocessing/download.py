"""Download helpers used by `scripts/*.py`."""

import os
import shutil
import sys
import urllib.request
from pathlib import Path


def dataset_dir(name: str) -> Path:
    """
    Return the dataset directory.

    Parameters
    ----------
    name : str
        Dataset subdirectory name to append to the base path.

    Returns
    -------
    Path
        The (created) dataset directory `base / name`.

    """
    home = os.environ.get("DATASETS_HOME")
    base = Path(home) if home else Path("datasets")
    out = base / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def download_if_missing(url: str, dest: Path) -> None:
    """
    Stream-download `url` to `dest`.

    Parameters
    ----------
    url : str
        Source URL to fetch.
    dest : Path
        Destination file path to write the downloaded content to.

    """
    if dest.exists():
        print(f"  found {dest.name} ({dest.stat().st_size / 1e6:.1f} MB) - skipping")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}\n  -> {dest}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as out:
        print(f"  HTTP/1.1 {r.status} {r.reason}\n{r.headers}", end="")
        shutil.copyfileobj(r, out)
    print(f"  done ({dest.stat().st_size / 1e6:.1f} MB)")


def kaggle_download_file_if_missing(
    dataset: str,
    version: int,
    file_name: str,
    dest: Path,
) -> None:
    """
    Use Kaggle CLI to download `{dataset}/{version}/{file_name}` into `dest`.

    Parameters
    ----------
    dataset : str
        Kaggle dataset slug (without the version component).
    version : int
        Dataset version number.
    file_name : str
        Name of the file to download from the dataset.
    dest : Path
        Destination directory the file is downloaded into.

    """
    target = dest / file_name
    if target.exists():
        print(
            f"  found {target.name} ({target.stat().st_size / 1e6:.1f} MB) - skipping"
        )
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  kaggle download {dataset}/{version} :: {file_name}\n  -> {dest}")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit(
            "  kaggle package not installed. Install with: uv sync --extra data\n"
            "  Then follow the instructions to authenticate at:\n"
            "  https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication"
        )

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(
        f"{dataset}/{version}",
        file_name=file_name,
        path=str(dest),
    )

    zipped = dest / f"{file_name}.zip"
    if zipped.exists():
        import zipfile

        with zipfile.ZipFile(zipped) as zf:
            zf.extractall(dest)
        zipped.unlink()


# File refactored by claude-opus-4-8 using prompt: Add pretty print statements to
# `scripts/amazon_books_2014.py`, `scripts/gowalla.py`, `scripts/yelp_2018.py`,
# `src/common/preprocessing/logging.py` and `src/common/preprocessing/download.py`. I
# want to see details of data provenance and vintage. Each step (download, load, dedupe,
# k-core, output) should be clearly identifiable, and each step should report only
# useful information. If a reference split and a diagnostic cannot be run, it should say
# so. Use plain `print` calls inside the existing functions; don't add new functions,
# constants or imports.
