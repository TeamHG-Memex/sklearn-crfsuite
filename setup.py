#!/usr/bin/env python
from setuptools import setup

setup(
    name='sklearn-crfsuite',
    version='0.3.6',
    author='Mikhail Korobov',
    author_email='kmike84@gmail.com',
    license='MIT license',
    long_description=open('README.rst').read() + "\n\n" + open('CHANGES.rst').read(),
    description="CRFsuite (python-crfsuite) wrapper which provides interface simlar to scikit-learn",
    url='https://github.com/TeamHG-Memex/sklearn-crfsuite',
    zip_safe=False,
    packages=['sklearn_crfsuite'],
    install_requires=[
        "tqdm >= 2.0",
        "six",
        "tabulate",
        "python-crfsuite >= 0.8.3"
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
