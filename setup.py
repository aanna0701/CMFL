#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, dist
dist.Distribution(dict(setup_requires=['bob.extension']))

from bob.extension.utils import load_requirements, find_packages
install_requires = load_requirements()



setup(

    name='bob.paper.cross_modal_focal_loss_cvpr2021',
    version=open("version.txt").read().rstrip(),
    description='Source code package for CVPR2021 paper Cross Modal Focal Loss for RGBD Face Anti-Spoofing',

    url='https://gitlab.idiap.ch/bob/bob.paper.cross_modal_focal_loss_cvpr2021',
    license='GPLv3',

    author='Anjith George',
    author_email='anjith.george@idiap.ch',

    keywords = "bob",

    long_description=open('README.rst').read(),

    # leave this here, it is pretty standard
    packages=find_packages(),
    include_package_data=True,
    zip_safe = False,

    install_requires=install_requires,

    entry_points={
    "dask.client": [
        "sge-big         = bob.paper.cross_modal_focal_loss_cvpr2021.distributed.sge_big:dask_client",
    ],
    },


    # check classifiers, add and remove as you see fit
    # full list here: https://pypi.org/classifiers/
    # don't remove the Bob framework unless it's not a bob package
    classifiers = [
      'Framework :: Bob',
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Software Development :: Libraries :: Python Modules',
      ],

)