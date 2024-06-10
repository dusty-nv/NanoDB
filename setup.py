#!/usr/bin/env python3
from setuptools import setup, find_packages
from nanodb.version import __version__

setup(
    name="nanodb",
    version=__version__,
    packages=find_packages()
)
