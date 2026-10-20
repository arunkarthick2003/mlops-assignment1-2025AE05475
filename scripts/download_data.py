"""Fetch and save the UCI Cleveland dataset as a cleaned CSV."""

from pathlib import Path

from heart_mlops.data import load_or_download


def main() -> None:
    raw_path = Path("data/raw/processed.cleveland.data")
    frame = load_or_download(raw_path)
    cleaned_path = Path("data/processed/heart_disease.csv")
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cleaned_path, index=False)
    print(f"Saved {len(frame)} rows and {len(frame.columns) - 1} features to {cleaned_path}")


if __name__ == "__main__":
    main()