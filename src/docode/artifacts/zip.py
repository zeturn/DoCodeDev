from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def zip_files(destination: Path, files: list[Path], root: Path) -> int:
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for file in files:
            if not file.exists() or not file.is_file():
                continue
            archive.write(file, file.relative_to(root))
    return destination.stat().st_size

