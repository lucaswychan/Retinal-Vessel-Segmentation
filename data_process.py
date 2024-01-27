import numpy as np


# Contrast Stretching
def contrast_stretch(x):
    """Stretch the contrast of each individual images in the given data array.

    Parameters
    ------------
    x : np.ndarray
        Image data array
    Returns:
    -------------
    np.ndarray
        New image data arary with individually contrast-stretched images.
    """

    I_min = np.expand_dims(np.min(x, axis=(1, 2)), axis=(1, 2))
    I_max = np.expand_dims(np.max(x, axis=(1, 2)), axis=(1, 2))
    x_enhanced = ((x - I_min) / (I_max - I_min)) * 255

    return x_enhanced


# Image Re-scaling
def rescale_01(x):
    """Rescales the given image data array to range [0,1].

    Parameters
    ------------
    x : np.ndarray
        image data array

    Returns:
    -------------
    np.ndarray
        New image data arary re-scaled to range [0,1].
    """

    x_01 = x.astype(float) / 255.0

    return x_01.astype(float)
