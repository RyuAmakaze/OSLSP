# Configuration constants used for training
DAY_PROP = [
    [0.5, 0.5, 0.0, 0.0],  # day0(tri, quad, penta, hexa)
    [0.0, 0.5, 0.5, 0.0],  # day3(tri, quad, penta, hexa)
    [0.33, 0.0, 0.33, 0.33],  # day5(tri, quad, penta, hexa)
    [0.4, 0.4, 0.2, 0.0],  # day7(tri, quad, penta, hexa)
    [0.1, 0.0, 0.1, 0.8],  # day14(tri, quad, penta, hexa)
]

CLASS_DIVER = [
    [1,   2/3,  1/3,  0],
    [2/3, 1,   2/3,  1/3],
    [1/3, 2/3,   1,  2/3],
    [0,   1/3,  2/3,  1],
]

GAUSSIAN_SIGMA = 0.1
RESIZE = 64
GLOBAL_CROP_SIZE = 224
LOCAL_CROP_SIZE = 96

NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)

# Data augmentation parameters
HORIZONTAL_FLIP_PROB = 0.5
COLOR_JITTER_BRIGHTNESS = 0.4
COLOR_JITTER_CONTRAST = 0.4
COLOR_JITTER_SATURATION = 0.2
COLOR_JITTER_HUE = 0.1
COLOR_JITTER_PROB = 0.8
RANDOM_GRAYSCALE_PROB = 0.2
GAUSSIAN_BLUR1_PROB = 1.0
GAUSSIAN_BLUR2_PROB = 0.1
GAUSSIAN_BLUR_LOCAL_PROB = 0.5
SOLARIZATION_PROB = 0.2

# MLP layer dimensions
MLP_HIDDEN_DIM_1 = 128
MLP_HIDDEN_DIM_2 = 64

# Validation settings
VALID_LOG_FREQ = 20
