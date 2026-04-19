"""CSV logging for human gameplay demonstrations."""

import csv
import os


class CSVDataLogger:
    """Append feature rows plus labels to a CSV file."""

    def __init__(self, path: str):
        self.path = path
        self.fieldnames: list[str] | None = None
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, features: dict[str, int], label: str):
        if self.fieldnames is None:
            self.fieldnames = list(features.keys()) + ["label"]

        row = dict(features)
        row["label"] = label

        file_exists = os.path.exists(self.path)
        needs_header = (not file_exists) or os.path.getsize(self.path) == 0

        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
