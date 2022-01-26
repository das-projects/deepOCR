import deepocr


def test_version():
    assert len(deepocr.__version__.split('.')) == 3


def test_is_torch_available():
    assert not deepocr.is_torch_available()
