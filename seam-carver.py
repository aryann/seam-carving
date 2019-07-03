import sys

import PIL
import numpy as np
from matplotlib import image
from matplotlib import pyplot


_RESIZE_FACTOR = 1

# Determines while energy levels are visualized. Any cell value that
# is below this percentile is displayed as black. Cell values at or
# above this percentile are displayed as white.
_ENERGY_PERCENTILE = 95


def compute_energy(data):
    """N.B.: energies for the edges are not computed."""
    result = np.zeros(data.shape, dtype=np.int32)

    for i in range(1, data.shape[0] - 1):
        for j in range(1, data.shape[1] - 1):
            left_rgb = data[i, j - 1]
            right_rgb = data[i, j + 1]
            horizontal = sum((int(left) - int(right)) ** 2 for left, right
                             in zip(left_rgb, right_rgb))

            top_rgb = data[i - 1, j]
            bottom_rgb = data[i + 1, j]
            vertical = sum((int(top) - int(bottom)) ** 2 for top, bottom
                           in zip(top_rgb, bottom_rgb))

            result[i, j] = horizontal + vertical

    return result[1:-1, 1:-1]


if __name__ == '__main__':
    image_obj = PIL.Image.open(sys.argv[1])
    image_obj.thumbnail((image_obj.size[0] * _RESIZE_FACTOR,
                         image_obj.size[1] * _RESIZE_FACTOR))

    data = np.array(image_obj)
    print(data.dtype)
    print(data.shape)

    pyplot.figure()
    pyplot.subplot(2, 1, 1)
    pyplot.imshow(data)

    np.pad(data, pad_width=1, mode='edge')

    energy = compute_energy(data)

    energy[energy < np.percentile(energy, _ENERGY_PERCENTILE)] = 0
    print(energy)
    pyplot.subplot(2, 1, 2)
    pyplot.imshow(energy)
    pyplot.show()
