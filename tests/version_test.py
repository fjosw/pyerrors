from importlib.metadata import version

import pyerrors as pe


def test_version_matches_package_metadata():
    assert pe.__version__ == version("pyerrors")
