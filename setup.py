#!/usr/bin/env python
from setuptools import setup

setup(
    name='sklearn-crfsuite',
    version='0.3.7',
    author='Mikhail Korobov',
    author_email='kmike84@gmail.com',
    license='MIT license',
    long_description=open('README.rst').read() + "\n\n" + open('CHANGES.rst').read(),
    description="CRFsuite (python-crfsuite) wrapper which provides interface simlar to scikit-learn",
    url='https://github.com/TeamHG-Memex/sklearn-crfsuite',
    zip_safe=False,
    packages=['sklearn_crfsuite'],
    python_requires='>=3.6',
    install_requires=[
        "python-crfsuite >= 0.9.6",
        "scikit-learn >= 0.20",
        "tabulate >= 0.8.3",
        "tqdm >= 4.29.0",
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
