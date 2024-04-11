"""Extra helper functions for extension."""

from pathlib import Path
import shutil
import warnings


def copy_files_to_dest(dest: str, files: list[str]) -> list[str]:
    """Copies files to destination directory.

    Args:
        dest: Destination to copy files to.
        files: List of files to copy.

    Returns:
        List of new filepaths after move.
    """
    target_dir = Path(dest)

    # noop if already exists
    target_dir.mkdir(exist_ok=True)

    def _move_file(file: str) -> str:
        f = Path(file)
        if not f.is_file():
            warnings.warn(f"{file} does not exist or is not a file")
            return ""
        return shutil.copy(f, target_dir)

    updated_files = {_move_file(f) for f in files}
    updated_files.discard("")
    return list(updated_files)


def delete_files_from_dest(dest: str, files: list[str]) -> list[str]:
    """Deletes files from destination directory.

    Args:
        dest: Destination to delete files from.
        files: List of files to delete.

    Returns:
        List of deleted files.
    """
    target_dir = Path(dest)
    if not target_dir.exists():
        raise Exception(
            f"Error [habitllm.utils.delete_files_from_dest]:, {dest} does not exist."
        )

    def _delete_valid_files(file: str) -> str:
        f = Path(file)
        if not (f.is_file() and f.resolve().parent.samefile(target_dir.resolve())):
            warnings.warn(f"{file} does not exist or is not in {dest}")
            return ""
        f.unlink(missing_ok=True)
        return str(f)

    deleted_files = {_delete_valid_files(f) for f in files}
    deleted_files.discard("")
    return list(deleted_files)
