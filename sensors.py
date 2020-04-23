POSITION_SENSOR_TYPES = [
    "Position_X",
    "Position_Y",
    "Position_Z",
]

ROTATION_SENSOR_TYPES = [
    "Rotation_X",
    "Rotation_Z",
    "Rotation_W",
    "Rotation_Y"
]

ALL_SENSOR_TYPES = POSITION_SENSOR_TYPES + ROTATION_SENSOR_TYPES

TOP_LEVEL_SENSORS = [
    "Chest", "Head"
]

EQUIVALENT_SENSORS = [
    ("Neck", "Head")
]

IGNORE_SENSORS = [
    "LToe", "RToe", "RShin", "LShin",
    "LThigh", "RThigh", "Hip", "Ab"
]

SENSOR_GROUPS = {
    "LFinger1": ("LIndex1", "LMiddle1", "LPinky1", "LRing1"),
    "LFinger2": ("LIndex2", "LMiddle2", "LPinky2", "LRing2"),
    "LFinger3": ("LIndex3", "LMiddle3", "LPinky3", "LRing3"),
    "RFinger1": ("RIndex1", "RMiddle1", "RPinky1", "RRing1"),
    "RFinger2": ("RIndex2", "RMiddle2", "RPinky2", "RRing2"),
    "RFinger3": ("RIndex3", "RMiddle3", "RPinky3", "RRing3")
}

SENSOR_REFERENCE = {
    "LThigh": "Chest",
    "RThigh": "Chest",
    "LShin": "Chest",
    "RShin": "Chest",
    "LIndex1": "LHand",
    "LIndex2": "LHand",
    "LIndex3": "LHand",
    'LMiddle1': "LHand",
    'LMiddle2': "LHand",
    'LMiddle3': "LHand",
    'LPinky1': "LHand",
    'LPinky2': "LHand",
    'LPinky3': "LHand",
    'LRing1': "LHand",
    'LRing2': "LHand",
    'LRing3': "LHand",
    'LThumb1': "LHand",
    'LThumb2': "LHand",
    'LThumb3': "LHand",
    "RIndex1": "RHand",
    "RIndex2": "RHand",
    "RIndex3": "RHand",
    'RMiddle1': "RHand",
    'RMiddle2': "RHand",
    'RMiddle3': "RHand",
    'RPinky1': "RHand",
    'RPinky2': "RHand",
    'RPinky3': "RHand",
    'RRing1': "RHand",
    'RRing2': "RHand",
    'RRing3': "RHand",
    'RThumb1': "RHand",
    'RThumb2': "RHand",
    'RThumb3': "RHand",
    "LHand": "Chest",
    "RHand": "Chest",
    "LFArm": "LShoulder",
    "RFArm": "RShoulder",
    "LUArm": "LShoulder",
    "RUArm": "RShoulder",
    "LShoulder": "Chest",
    "RShoulder": "Chest",
    "Neck": "Chest",
    "LFoot": "Chest",
    "RFoot": "Chest"
}