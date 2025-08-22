from pathlib import Path


def count_files_in_directory(directory):
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        return 0
    return len([f for f in dir_path.iterdir() if f.is_file()])
