import platform


def adapt_path(path: str) -> str:
    """
    Adapt path to Windows systems if necessary.
    """
    if platform.system() == "Windows":
        return path.replace("/", "\\")
    return path
