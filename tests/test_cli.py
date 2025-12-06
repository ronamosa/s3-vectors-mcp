import json
from s3_vectors_mcp.cli import main


def test_version_flag(capsys):
    exit_code = main(["--version"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip()

