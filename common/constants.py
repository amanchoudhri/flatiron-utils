from pathlib import Path

import numpy as np

PRA_ARRAYS_PATH = Path('/mnt/home/atanelus/pra_arrays/')

LARGE_MIC_POS = np.load(PRA_ARRAYS_PATH / 'large_mic_pos.npy')
LARGE_ROOM_DIMS = np.load(PRA_ARRAYS_PATH / 'large_room_dims.npy')

# by default, pyroomacoustics sets the temperature to be
# such that the speed of sound is 343 m/s
# as per:
# https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html#pyroomacoustics.room.ShoeBox
from pyroomacoustics.parameters import Physics
TEMP = Physics.from_speed(343).T  # T param is where temp is stored in the Physics object

SAMPLE_RATE = 125000  # all our audio is sampled at 125 kHz

# default values for MUSE frequency filtering
F_LO = 0
F_HI = 62500