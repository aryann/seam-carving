import logging
import sys

import PIL
import numpy as np
from matplotlib import animation
from matplotlib import image
from matplotlib import pyplot


_RESIZE_FACTOR = .2

# Determines while energy levels are visualized. Any cell value that
# is below this percentile is displayed as black. Cell values at or
# above this percentile are displayed as white.
_ENERGY_PERCENTILE = 95


def compute_energy(data):
    """N.B.: energies for the edges are not computed."""
    result = np.zeros(data.shape[:2], dtype=np.int32)

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


def compute_seam_costs(energy):
    """Computes the seam costs for the given image energy matrix. Seams
    are computed from the top row to the bottom row. This function
    returns two matrices:

      (1) A matrix that contains seam costs for all cells. For each
          cell, the cost is the energy at that cell + the minimum cost
          of the closest three cells in the row above the cell.

      (2) A matrix that contains the indices required to traverse each
          seam. For each cell, the value contains the index of the
          cell in the above row that led to the minimum seam that
          terminates at that cell.

    """
    costs = np.full((energy.shape[0], energy.shape[1] + 2),
                    fill_value=np.iinfo(np.int64).max,
                    dtype=np.int64)
    indices = np.zeros(costs.shape, dtype=np.int32)

    for j in range(1, costs.shape[1] - 1):
        costs[0, j] = energy[0, j - 1]
        indices[0, j] = -1

    for i in range(1, costs.shape[0]):
        for j in range(1, costs.shape[1] - 1):
            items = costs[i - 1, j - 1 : j + 2]
            costs[i, j] = energy[i, j - 1] + items.min()
            indices[i, j] = j + items.argmin() - 2

    return costs[:, 1:-1], indices[:, 1:-1]


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    image_obj = PIL.Image.open(sys.argv[1])
    image_obj.thumbnail((image_obj.size[0] * _RESIZE_FACTOR,
                         image_obj.size[1] * _RESIZE_FACTOR))

    data = np.array(image_obj)
    logging.info('Image shape: %s', data.shape)

    np.pad(data, pad_width=1, mode='edge')

    energy = compute_energy(data)
    logging.info('Finished computing image energy: %s', energy)
    logging.info('Finished computing seam costs: %s', compute_seam_costs(energy))
    energy[energy < np.percentile(energy, _ENERGY_PERCENTILE)] = 0

    figure = pyplot.figure()

    pyplot.subplot(2, 1, 1).axis('off')
    pyplot.imshow(energy)

    pyplot.subplot(2, 1, 2).axis('off')
    curr = pyplot.imshow(energy)

    def animate(i):
        # TODO(aryann): Use this function to run one iteration of the
        # seam carving algorithm.
        if i % 2 == 0:
            curr.set_data(data)
        else:
            curr.set_data(energy)

    _ = animation.FuncAnimation(figure, animate, interval=1000)
    pyplot.show()
