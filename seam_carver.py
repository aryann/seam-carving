import collections
import logging
import threading
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

_SEAM_COLOR = (255, 255, 255)


Plots = collections.namedtuple(
    'Plots',
    ['energy', 'image_with_seam', 'cropped_image'])


def compute_energy(data):
    """N.B.: energies for the edges are not computed."""
    data = np.pad(data, pad_width=1, mode='edge')

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

    costs[0, 1 : costs.shape[1] - 1] = energy[0, :]
    indices[0, :] = -1

    for i in range(1, costs.shape[0]):
        for j in range(1, costs.shape[1] - 1):
            items = costs[i - 1, j - 1 : j + 2]
            costs[i, j] = energy[i, j - 1] + items.min()
            indices[i, j] = j + items.argmin() - 2

    return costs[:, 1:-1], indices[:, 1:-1]


def get_min_seam_indices(costs, indices):
    result = np.zeros(costs.shape[0], dtype=np.int32)
    result[0] = costs[-1, :].argmin()
    for i in range(1, result.size):
        result[i] = indices[indices.shape[0] - i, result[0]]

    return np.flip(result)


def remove_seam(image, seam_indices):
    result = np.zeros((image.shape[0], image.shape[1] - 1, image.shape[2]),
                      dtype=np.int64)
    for i in range(image.shape[0]):
        result[i] = np.concatenate((image[i, :seam_indices[i], :],
                                   image[i, seam_indices[i] + 1:, :]))
    return result


def color_seam(data, seam_indices):
    result = data.copy()
    for i in range(data.shape[0]):
        result[i, seam_indices[i]] = _SEAM_COLOR
    return result


def add_border_for_rendering(data, width):
    return np.concatenate((
        data,
        np.ones(shape=(
            data.shape[0], width, data.shape[2]), dtype=np.int64) * (255, 0, 0)),
                          axis=1)


def run_iteration(data, plots, num_iterations):
    logging.info('Starting iteration %d.', num_iterations)

    energy = compute_energy(data)
    logging.debug('Finished computing image energy:\n%s', energy)
    logging.debug('Energy shape: %s', energy.shape)
    plots.energy.set_data(energy)

    costs, indices = compute_seam_costs(energy)
    logging.debug('Finished computing seam costs and indices:\n%s\n\n%s',
                  costs, indices)

    min_seam_indices = get_min_seam_indices(costs, indices)
    logging.debug('Min seam indices:\n%s', min_seam_indices)

    plots.image_with_seam.set_data(color_seam(data, min_seam_indices))

    data = remove_seam(data, min_seam_indices)
    plots.cropped_image.set_data(add_border_for_rendering(data, num_iterations))
    logging.info('Done with iteration..')
    return data


def run(data, plots):
    num_iterations = 0
    while True:
        num_iterations += 1
        data = run_iteration(data, plots, num_iterations)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    image_obj = PIL.Image.open(sys.argv[1])
    image_obj.thumbnail((image_obj.size[0] * _RESIZE_FACTOR,
                         image_obj.size[1] * _RESIZE_FACTOR))

    data = np.array(image_obj)

    figure = pyplot.figure()

    energy_plot = pyplot.subplot(2, 2, 1)
    energy_plot.set_title('energy')
    energy_plot.axis('off')
    energy_image = pyplot.imshow(data)
    energy_image.set_clim(vmin=0, vmax=6 * ((1 << 8) - 1) ** 2)

    original_image_plot = pyplot.subplot(2, 2, 2)
    original_image_plot.set_title('original')
    original_image_plot.axis('off')
    pyplot.imshow(data)

    image_with_seam_plot = pyplot.subplot(2, 2, 3)
    image_with_seam_plot.set_title('current seam')
    image_with_seam_plot.axis('off')
    image_with_seam = pyplot.imshow(data)

    cropped_image_plot = pyplot.subplot(2, 2, 4)
    cropped_image_plot.set_title('cropped image')
    cropped_image_plot.axis('off')
    cropped_image = pyplot.imshow(data)

    plots = Plots(
        energy=energy_image,
        image_with_seam=image_with_seam,
        cropped_image=cropped_image)
    threading.Thread(target=run, args=(data, plots)).start()

    def animate(i, plots):
        pass

    _ = animation.FuncAnimation(figure, animate, fargs=(plots,), interval=100)
    pyplot.show()
