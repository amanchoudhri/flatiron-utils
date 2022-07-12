from pathlib import Path

import numpy as np

PRA_ARRAYS_PATH = Path('/mnt/home/atanelus/pra_arrays/')

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

# ====== SIMULATED DATA SCENARIOS ======

def room_info(mic_pos, room_dims):
    DIMS_ARR = np.load(PRA_ARRAYS_PATH / (room_dims + '.npy'))
    POS_ARR = np.load(PRA_ARRAYS_PATH / (mic_pos + '.npy'))
    return {'mic_pos': POS_ARR, 'room_dims': DIMS_ARR}

SCENARIOS = {
    'small_room_4': room_info('small_mic_pos', 'small_room_dims'),
    'big_room_8ceiling': room_info('mic8_ceiling', 'large_room_dims'),
    'big_room_16': room_info('mic16_ceiling', 'large_room_dims'),
    'big_room_8floor': room_info('mic8_floor', 'large_room_dims'),
}

# ======================================