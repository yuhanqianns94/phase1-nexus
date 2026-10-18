"""The public entry point has no source mixing, configurable cap, or history offset."""
import pytest
from phase1_acquisition.cli import main


@pytest.mark.parametrize('args', [['discover'], ['run','--sources','ctgov'], ['run','--limit','50'],
                                  ['run','--per-source','400'], ['history-import']])
def test_removed_modes_are_rejected(args):
    with pytest.raises(SystemExit) as e:
        main(args)
    assert e.value.code == 2


def test_status_is_read_only(tmp_path, capsys):
    main(['status','--workspace',str(tmp_path)])
    assert 'not_started' in capsys.readouterr().out
    assert not list(tmp_path.iterdir())
