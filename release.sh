#!/usr/bin/env bash
python3 setup.py sdist bdist_wheel
twine upload dist/*

rm -rf build
rm -rf dist
rm -rf *.egg-info
