import os
from typing import List, Optional

from beartype import beartype


@beartype
def save_text_to_file(some_text: str, path: str, overwrite: bool = True):
    """
    Save text to a file. If there is no file ending, ".txt" is added. By default, the file is overwritten if it
    already exists.

    Args:
        some_text (str): The text that should be saved to the file.
        path (str): The path to the output file.
        overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to True.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # check if a file ending is present
    if os.path.splitext(path)[1] == "":
        path = path + ".txt"

    if not overwrite:
        # check if file already exists
        counter = 1
        base_path, file_ext = os.path.splitext(path)
        while os.path.isfile(path):
            # file exists, so rename it
            path = f"{base_path}_{counter}{file_ext}"
            counter += 1

    with open(path, "w") as f:
        f.write(some_text)


@beartype
def find_file(dir_path: str, possible_filenames: List[str]) -> Optional[str]:
    for filename in possible_filenames:
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            return file_path
    return None


@beartype
def find_first_file_with_extension(dir_path: str, file_extension) -> Optional[str]:
    for filename in os.listdir(dir_path):
        if filename.endswith(file_extension):
            return os.path.join(dir_path, filename)
    return None
