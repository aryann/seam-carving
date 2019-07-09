import logging
import math
import sys

import PIL
import numpy as np
from matplotlib import pyplot


_RESIZE_FACTOR = .2


def compute_energy(data):
    """N.B.: energies for the edges are not computed."""
    data = np.pad(data, pad_width=1, mode='edge')

    result = np.empty(data.shape[:2], dtype=np.int32)

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

            result[i, j] = math.sqrt(horizontal + vertical)

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
    indices = np.empty(costs.shape, dtype=np.int32)

    costs[0, 1 : costs.shape[1] - 1] = energy[0, :]
    indices[0, :] = -1

    for i in range(1, costs.shape[0]):
        for j in range(1, costs.shape[1] - 1):
            items = costs[i - 1, j - 1 : j + 2]
            costs[i, j] = energy[i, j - 1] + items.min()
            indices[i, j] = j + items.argmin() - 2

    return costs[:, 1:-1], indices[:, 1:-1]


def get_min_seam_indices(costs, indices):
    result = np.empty(costs.shape[0], dtype=np.int32)
    result[0] = costs[-1, :].argmin()
    for i in range(1, result.size):
        result[i] = indices[indices.shape[0] - i, result[0]]

    return np.flip(result)


def remove_seam(image, seam_indices):
    result = np.empty((image.shape[0], image.shape[1] - 1, image.shape[2]),
                      dtype=np.int64)
    for i in range(image.shape[0]):
        result[i] = np.concatenate((image[i, :seam_indices[i], :],
                                   image[i, seam_indices[i] + 1:, :]))
    return result


def add_border_for_rendering(data, width):
    return np.concatenate((
        data,
        np.ones(shape=(
            data.shape[0], width, data.shape[2]), dtype=np.int64) * (255, 0, 0)),
                          axis=1)


if __name__ == '__main__':
    filepath = sys.argv[1]
    num_iterations = int(sys.argv[2])

    logging.getLogger().setLevel(logging.INFO)
    image_obj = PIL.Image.open(sys.argv[1])
    image_obj.thumbnail((image_obj.size[0] * _RESIZE_FACTOR,
                         image_obj.size[1] * _RESIZE_FACTOR))

    figure = pyplot.figure()
    data = np.array(image_obj)

    original_plot= pyplot.subplot(2, 2, 1)
    original_plot.set_title('original')
    original_plot.axis('off')
    pyplot.imshow(data)

    energy = compute_energy(data)
    energy_plot = pyplot.subplot(2, 2, 2)
    energy_plot.set_title('energy')
    energy_plot.axis('off')
    pyplot.imshow(energy)

    costs, indices = compute_seam_costs(energy)
    costs_plot = pyplot.subplot(2, 2, 4)
    costs_plot.set_title('seam costs')
    costs_plot.axis('off')
    pyplot.imshow(costs)

    for i in range(num_iterations):
        logging.info('Iteration: %d', i)
        min_seam_indices = get_min_seam_indices(costs, indices)
        data = remove_seam(data, min_seam_indices)
        energy = compute_energy(data)
        costs, indices = compute_seam_costs(energy)

    shrunk_plot = pyplot.subplot(2, 2, 3)
    shrunk_plot.set_title('shrunk')
    shrunk_plot.axis('off')
    pyplot.imshow(add_border_for_rendering(data, num_iterations))

    pyplot.show()
