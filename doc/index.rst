.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Mon 13 Aug 2012 12:36:40 CEST

.. _bob.paper.cross_modal_focal_loss_cvpr2021:

==================================================
Cross Modal Focal Loss for RGBD Face Anti-Spoofing
==================================================

This package is part of the Bob_ toolkit and it allows to reproduce the experimental results published in the following paper::

    @inproceedings{GeorgeCVPR2021,
        author = {Anjith George, Sebastien Marcel},
        title = {Cross Modal Focal Loss for RGBD Face Anti-Spoofing},
        year = {2021},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    }

If you use this package and/or its results, please cite the paper [GM21]_.


The main idea in the paper the multi-head architecture and the use of the proposed Cross-Modal Focal Loss
to improve the performance of multi-channel CNN models.

.. figure:: img/Framework2.png
   :align: center

   Diagram of the proposed framework (The architecture is referred to as RGBD-MH).


User guide
---------------------

.. toctree::
   :maxdepth: 2

   running_cmfl_hqwmca
   references


.. include:: links.rst

