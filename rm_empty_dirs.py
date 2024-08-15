import argparse
from pathlib import Path


def remove_empty_folders(path: Path):
    # Check if the path is a directory
    if not path.is_dir():
        return

    # Recursively remove empty folders from all subdirectories
    for item in path.iterdir():
        if item.is_dir():
            remove_empty_folders(item)

    # If the directory is empty after removing subdirectories, remove it
    if not any(path.iterdir()):
        path.rmdir()
        print(f"Removed empty folder: {path}")


def main():
    # Set up argparse to handle command-line arguments
    parser = argparse.ArgumentParser(
        description="Recursively remove empty folders from a parent folder."
    )
    parser.add_argument("parent_folder", type=str, help="The path to the parent folder")

    args = parser.parse_args()

    # Convert the provided path to a Path object
    parent_folder = Path(args.parent_folder)

    # Remove empty folders from the parent folder
    remove_empty_folders(parent_folder)


if __name__ == "__main__":
    main()
