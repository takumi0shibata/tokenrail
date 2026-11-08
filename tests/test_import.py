import subprocess
import sys
from pathlib import Path


def test_source_import_without_distribution_metadata():
    source_path = Path(__file__).resolve().parents[1] / "src"
    script = """
import sys
from importlib import metadata
from unittest.mock import patch

sys.path.insert(0, sys.argv[1])
installed_version = metadata.version


def version(name):
    if name == "tokenrail":
        raise metadata.PackageNotFoundError(name)
    return installed_version(name)


with patch.object(metadata, "version", version):
    from tokenrail import BatchExecutor, RailClient, __version__

assert BatchExecutor.__name__ == "BatchExecutor"
assert RailClient.__name__ == "RailClient"
assert __version__ == "0+unknown"
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(source_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
