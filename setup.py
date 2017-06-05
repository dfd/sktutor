#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os
import sys

base_dir = os.path.dirname(__file__)
src_dir = os.path.join(base_dir, "src")

sys.path.insert(0, src_dir)

about = {}
with open(os.path.join(src_dir, "sktutor", "__about__.py")) as f:
    exec(f.read(), about)

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
        'scipy>=0.19.0',
        'pandas>=0.20.1',
        'scikit-learn>=0.18.1'
]

test_requirements = [
        'pytest'
]

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__summary__'],
    long_description=readme + '\n\n' + history,
    author=about['__author__'],
    author_email=about['__email__'],
    url=about['__uri__'],
    license=about['__license__'],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='sktutor',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
