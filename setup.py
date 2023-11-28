#!/usr/bin/env python3
'''Installs dmri_pcconv'''

from os import path
from setuptools import setup, find_namespace_packages

install_deps = ['torch', 'lightning', 'npy-patcher', 'einops', 'nibabel']

version = '1.0.0'
this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dmri-pcconv',
    version=version,
    description='Parametric Continuous Convolution framework used for Diffusion MRI.',
    author='Matthew Lyon',
    author_email='matthewlyon18@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
    license='MIT License',
    packages=find_namespace_packages(),
    install_requires=install_deps,
    scripts=[],
    classifiers=[
        'Programming Language :: Python',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    keywords=['ai', 'cv', 'computer-vision', 'mri', 'dmri', 'super-resolution', 'cnn', 'pcconv'],
)
