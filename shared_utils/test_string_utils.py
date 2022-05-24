import pytest
from string_utils import get_trailing_digits

def test_get_trailing_digits(): 
    with pytest.raises(ValueError):
        get_trailing_digits(list())

    assert get_trailing_digits('bronze123') == '123', 'Trailing Digits not returned.'

    assert get_trailing_digits('bro00nze123') == '123', 'Non-trailing digits may be returned.'

    assert get_trailing_digits('bronze') == '', 'Non-empty result returned with no trailing digits.'

if __name__ == '__main__': 
    retcode = pytest.main([__file__])