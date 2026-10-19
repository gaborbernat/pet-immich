from pathlib import Path

import pytest

from pet_immich.cli import main


@pytest.mark.parametrize(
    "arguments", [[], ["import"], ["export-encoder"]], ids=["missing-command", "missing-import-paths", "missing-source"]
)
def test_cli_requires_inputs(arguments: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["pet-immich", *arguments, "--state", str(tmp_path)])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2
