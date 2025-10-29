from pathlib import Path
import shutil
import kagglehub
from kagglehub.exceptions import KaggleApiHTTPError

dest_dir = Path("./data")
dest_dir.mkdir(parents=True, exist_ok=True)

try:
    downloaded_path = kagglehub.dataset_download("palbha/cmapss-jet-engine-simulated-data")
except KaggleApiHTTPError as e:
    print("Error descargando dataset:", e)
    raise

downloaded = Path(downloaded_path)

if downloaded.is_dir():
    for item in downloaded.iterdir():
        target = dest_dir / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))
    try:
        downloaded.rmdir()
    except OSError:
        pass
else:
    target = dest_dir / downloaded.name
    if target.exists():
        target.unlink()
    shutil.move(str(downloaded), str(target))

print("Path to dataset files:", str(dest_dir))
