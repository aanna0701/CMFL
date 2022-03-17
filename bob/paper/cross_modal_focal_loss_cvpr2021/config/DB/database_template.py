#!/usr/bin/env python
"""
HQWMCA Db is a database for face PAD experiments.
"""
from bob.io.stream import Stream
from bob.pad.base.pipelines.vanilla_pad import DatabaseConnector


from sklearn.preprocessing import Normalizer
import bob.core
logger = bob.core.log.setup("bob.learn.pytorch")
import numpy as np

import bob.ip.stereo

color = Stream('color')

intel_depth = Stream('intel_depth').adjust(color).warp(color)



streams = { 'color'     : color,
            'depth'     : intel_depth}

# *****

PROTOCOL = 'grand_test-curated' # define protocol name here
# # protocols=['grand_test-curated','LOO_Flexiblemask', 'LOO_Glasses', 'LOO_Makeup', 'LOO_Mannequin', 'LOO_Papermask','LOO_Print', 'LOO_Rigidmask', 'LOO_Tattoo','LOO_Replay']

ANNOTATIONS_DIR = {{ANNOTATIONS_DIR}} # specify

from bob.extension import rc as _rc
from bob.paper.cross_modal_focal_loss_cvpr2021.database import HQWMCAPadDatabase
database = DatabaseConnector(HQWMCAPadDatabase(protocol=PROTOCOL,
                             original_directory=_rc['bob.db.hqwmca.directory'],
                             original_extension='.h5',
                             annotations_dir = ANNOTATIONS_DIR,
                             streams=streams,
                             n_frames=10))


protocol = PROTOCOL


groups = ["train", "dev", "eval"]

"""The default groups to use for reproducing the baselines.
"""