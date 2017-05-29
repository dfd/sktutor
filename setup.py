#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

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
    name='sktutor',
    version='0.1.5',
    description="sktutor helps your machines learn.",
    long_description=readme + '\n\n' + history,
    author="Dave Decker",
    author_email='dave.decker@gmail.com',
    url='https://github.com/dfd/sktutor',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=requirements,
    license="BSD license",
    zip_safe=False,
    keywords='sktutor',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
