import argparse
import json
import sys
from pathlib import Path


def notebook_to_script(notebook_path, output_path=None):
    """Converts a Jupyter Notebook to a Python script."""

    with open(notebook_path, "r") as f:
        notebook = json.load(f)

    script = ""
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            script += "# %%\n"
            script += "".join(cell["source"]) + "\n\n"
        elif cell["cell_type"] == "markdown":
            script += "# %% [markdown]\n"
            for line in cell["source"]:
                script += "# " + line  # Add the '#' comment prefix here
            script += "\n\n"  # Add extra newline for better readability

    if output_path:
        with open(output_path, "w") as f:
            f.write(script)
    else:
        print(script)


def my_cat(files):
    for file in files:
        file = Path(file)
        print(f'<file name="{file.name}">')
        if file.suffix == ".ipynb":
            notebook_to_script(file)
        else:
            print(file.read_text())
        print("</file>")
        print("\n" * 4)


if (
    __name__ == "__main__"
):  # Crucial: This ensures the following runs only when executed directly
    if not sys.stdin.isatty():  # Check if input is being piped
        files = [line.strip() for line in sys.stdin]
    else:  # If no piped input, use argparse for command-line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("files", nargs="*")
        args = parser.parse_args()
        files = args.files

    my_cat(files)
