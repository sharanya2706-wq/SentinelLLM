import pandas as pd
from pathlib import Path


REQUIRED_COLUMNS = [
    "id",
    "category",
    "prompt",
    "expected_answer",
    "evaluation_criteria",
    "difficulty",
]


def load_benchmark(file_path):
    """
    Load and validate a SentinelLLM benchmark dataset.

    Supports both:
    - Default SentinelLLM benchmark.csv
    - User-uploaded CSV datasets
    """

    file_path = Path(file_path)

    # Check whether the file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # Check that the file is a CSV
    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            "Invalid file format. Please upload a CSV file."
        )

    try:
        dataset = pd.read_csv(file_path)

    except Exception as e:
        raise ValueError(
            f"Unable to read CSV file: {e}"
        )

    # Remove completely empty rows
    dataset = dataset.dropna(how="all")

    # Remove spaces from column names
    dataset.columns = (
        dataset.columns
        .astype(str)
        .str.strip()
    )

    # Check required columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    # Dataset should not be empty
    if dataset.empty:
        raise ValueError(
            "The uploaded dataset is empty."
        )

    # Check for empty IDs
    if dataset["id"].isnull().any():
        raise ValueError(
            "Some test cases are missing an id."
        )

    # Check for duplicate IDs
    duplicate_ids = dataset[
        dataset["id"].duplicated()
    ]["id"].tolist()

    if duplicate_ids:
        raise ValueError(
            "Duplicate test case IDs found: "
            f"{duplicate_ids}"
        )

    # Check for empty categories
    if dataset["category"].isnull().any():
        raise ValueError(
            "Some test cases are missing a category."
        )

    # Check for empty prompts
    if dataset["prompt"].isnull().any():
        raise ValueError(
            "Some test cases are missing a prompt."
        )

    # Clean text columns
    text_columns = [
        "id",
        "category",
        "prompt",
        "expected_answer",
        "evaluation_criteria",
        "difficulty",
    ]

    for column in text_columns:

        dataset[column] = (
            dataset[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return dataset


def show_dataset_summary(dataset):
    """
    Display basic information about a benchmark dataset.
    """

    print("\n===== SentinelLLM Dataset Summary =====")

    print(
        f"Total test cases: {len(dataset)}"
    )

    print("\nTests by category:")

    print(
        dataset["category"]
        .value_counts()
        .to_string()
    )

    print("\nTests by difficulty:")

    print(
        dataset["difficulty"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":

    dataset_path = "data/benchmark.csv"

    try:

        benchmark = load_benchmark(
            dataset_path
        )

        show_dataset_summary(
            benchmark
        )

    except Exception as e:

        print(
            f"\nDataset Error: {e}"
        )