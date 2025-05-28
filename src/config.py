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
