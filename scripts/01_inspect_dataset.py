"""Inspect the official FraudZen data without changing any records.

From the SIMGuard project root:
    python scripts/01_inspect_dataset.py
    python scripts/01_inspect_dataset.py --dataset Fraudzen-dataset
    python scripts/01_inspect_dataset.py --dataset Fraudzen-dataset.zip

Uses only Python's standard library. Does not extract ZIPs or train a model.
"""

import argparse
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

REQUIRED = (
    "fraudulentUsers.txt",
    "Op_1_CDRTrace.txt",
    "International_CDRTrace.txt",
    "softswitch_canceled_calls.txt",
)


def find_dataset():
    """Try the most common locations for an already extracted dataset."""
    candidates = (
        Path("data/raw/Fraudzen-dataset"),
        Path("data/raw/FraudZen-dataset"),
        Path("Fraudzen-dataset"),
        Path("FraudZen-dataset"),
        Path("data/raw"),
    )
    for candidate in candidates:
        if (candidate / "FraudZen_Cdrs").is_dir():
            return candidate
    raise FileNotFoundError(
        "Dataset not found. Provide the extracted folder or ZIP with "
        "--dataset PATH (e.g. --dataset data/raw/Fraudzen-dataset)."
    )


class DatasetView:
    """Read files from an extracted folder or ZIP without extracting anything."""

    def __init__(self, location):
        self.location = Path(location)
        self.zip_file = None
        if self.location.is_file() and self.location.suffix.lower() == ".zip":
            self.zip_file = ZipFile(self.location)
            names = [
                n for n in self.zip_file.namelist()
                if not n.endswith("/") and not n.startswith("__MACOSX/")
            ]
            # Strip the archive's outer directory, if present.
            candidate = next(
                (n for n in names if "/FraudZen_Cdrs/" in n),
                None,
            )
            if candidate:
                prefix = candidate.split("FraudZen_Cdrs/")[0]
            elif any(n.startswith("FraudZen_Cdrs/") for n in names):
                prefix = ""
            else:
                raise ValueError("ZIP does not contain FraudZen_Cdrs/")
            self.paths = {
                n[len(prefix):]: n for n in names
                if n.startswith(prefix) and not Path(n).name.startswith(".")
            }
        elif self.location.is_dir():
            if not (self.location / "FraudZen_Cdrs").is_dir():
                raise ValueError("Directory does not contain FraudZen_Cdrs/")
            self.paths = {
                p.relative_to(self.location).as_posix(): p
                for p in self.location.rglob("*")
                if p.is_file() and not p.name.startswith(".")
                and "__MACOSX" not in p.parts
            }
        else:
            raise FileNotFoundError(self.location)

    def read_text(self, relative_path):
        path = self.paths[relative_path]
        if self.zip_file:
            return self.zip_file.read(path).decode("utf-8", errors="replace")
        return path.read_text(encoding="utf-8", errors="replace")

    def first_lines(self, relative_path, count=2):
        path = self.paths[relative_path]
        if self.zip_file:
            handle = self.zip_file.open(path)
        else:
            handle = path.open("rb")
        with handle:
            return [handle.readline().decode("utf-8", errors="replace").strip()
                    for _ in range(count)]

    def close(self):
        if self.zip_file:
            self.zip_file.close()


def inspect(location):
    view = DatasetView(location)
    try:
        scenario_dirs = sorted({
            name.rsplit("/", 1)[0]
            for name in view.paths
            if name.startswith("FraudZen_Cdrs/")
            and name.endswith("/fraudulentUsers.txt")
        })
        print(f"Dataset location: {location}")
        print(f"Non-hidden files: {len(view.paths):,}")
        print(f"Simulation scenarios: {len(scenario_dirs)}")
        if not scenario_dirs:
            raise ValueError("No scenario label files found")

        print("\nScenarios and ground-truth fraudulent subscriber counts:")
        group_counts = Counter()
        total_labels = 0
        for scenario_dir in scenario_dirs:
            strategy = scenario_dir.split("/")[1]
            missing = [name for name in REQUIRED
                       if f"{scenario_dir}/{name}" not in view.paths]
            if missing:
                print(f"  WARNING: {scenario_dir}: missing {missing}")
            labels = set(view.read_text(f"{scenario_dir}/fraudulentUsers.txt").split())
            total_labels += len(labels)
            group_counts[strategy] += 1
            print(f"  {scenario_dir.removeprefix('FraudZen_Cdrs/')} => {len(labels)} fraud IDs")

        print("\nStrategy group counts:")
        for group, count in sorted(group_counts.items()):
            print(f"  {group}: {count} scenarios")
        print(f"Total fraud-label entries across scenarios: {total_labels:,}")
        print("Note: these are per-scenario labels, not unique subscribers across the dataset.")

        preferred = next(
            (p for p in scenario_dirs if p.startswith("FraudZen_Cdrs/all_advanced/")),
            scenario_dirs[0],
        )
        print(f"\nExample scenario: {preferred}")
        for filename in REQUIRED[1:]:
            rel = f"{preferred}/{filename}"
            if rel not in view.paths:
                continue
            header, first_record = view.first_lines(rel)
            headers = header.split()
            tokens = first_record.split()
            print(f"\n  {filename}")
            print(f"  Header columns ({len(headers)}): {', '.join(headers)}")
            print(f"  First record tokens: {len(tokens)}")
            if first_record:
                print(f"  First record type: {tokens[2] if filename != 'softswitch_canceled_calls.txt' and len(tokens) > 2 else 'N/A'}")
            else:
                print("  First record: empty (possible zero events)")

        print("\nInspection finished. No data has been modified.")
        print("NEXT: agree on parsing, features and scenario-level splits before training.")
    finally:
        view.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect official FraudZen scenarios")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Path to extracted FraudZen folder or uploaded ZIP")
    args = parser.parse_args()
    inspect(args.dataset if args.dataset else find_dataset())
