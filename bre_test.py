from pathlib import Path
import csv
import json
import yaml
import pandas as pd

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

from src.utils.get_folders_utils import get_dataset_path

path = Path(get_dataset_path(config))

with path.open("r", encoding="utf-8-sig", newline="") as f:
    headers = next(csv.reader(f))

with open("configs/BRE_selected_features.txt") as f:
    selected = [line.strip() for line in f if line.strip()]

sample = pd.read_csv(path, nrows=1000, keep_default_na=False)
sample_path = Path("BRE_first_1000_rows.csv")
sample.to_csv(sample_path, index=False)

print(json.dumps({
    "source": str(path),
    "bytes": path.stat().st_size,
    "headers": headers,
    "selected_count": len(selected),
    "selected_missing_from_header": [
        name for name in selected if name not in headers
    ],
    "sample_file": str(sample_path.resolve()),
}, indent=2))