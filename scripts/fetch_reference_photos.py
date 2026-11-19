"""Install only the private reference JPEGs listed in the checked-in manifest."""

import io
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

LIMIT = 32 * 1024 * 1024


def main():
    url = os.environ.get("CAD_PHOTOS_ZIP_URL", "")
    if urlsplit(url).scheme != "https":
        raise SystemExit("CAD_PHOTOS_ZIP_URL must be configured as an HTTPS URL")
    manifest = json.loads(Path("refs/calibration/upright.json").read_text())
    photos = Path("refs/photos")
    photos.mkdir(exist_ok=True)
    try:
        with urlopen(url, timeout=45) as response:
            data = response.read(LIMIT + 1)
        if len(data) > LIMIT:
            raise ValueError("Archive exceeds size limit")
        with ZipFile(io.BytesIO(data)) as archive:
            for view in manifest["views"]:
                name = view["photo"]
                entry = archive.getinfo(name)
                if entry.file_size > LIMIT:
                    raise ValueError("JPEG exceeds size limit")
                (photos / name).write_bytes(archive.read(entry))
    except (OSError, BadZipFile, ValueError, KeyError, RuntimeError):
        raise SystemExit("Could not load the ZIP or one of the required root-level JPEGs") from None


if __name__ == "__main__":
    main()
