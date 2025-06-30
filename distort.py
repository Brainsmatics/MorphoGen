import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from mpl_toolkits.mplot3d import Axes3D
import random

def add_block_noise(branches, num_blocks=3, noise_scale=0.2, block_size_range=(3, 8)):
    distorted = branches.copy()
    n_branches, n_points, _ = branches.shape
    for i in range(n_branches):
        num_blocks = random.randint(1,num_blocks)
        noise = np.zeros((n_points, 3))
        for _ in range(num_blocks):

            block_start = np.random.randint(1, n_points - block_size_range[1] - 1)
            block_size = np.random.randint(block_size_range[0], block_size_range[1] + 1)
            block_end = min(block_start + block_size, n_points - 1)
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            amplitude = noise_scale * np.ptp(branches[i], axis=0).mean()
            noise[block_start:block_end] += direction * amplitude
        noise_mean = noise[1:-1].mean(axis=0)
        noise[1:-1] -= noise_mean
        noise[0] = 0
        noise[-1] = 0

        distorted[i] += noise

    return distorted


def visualize_deformation(original, distorted, n_samples=8):
    indices = np.random.choice(len(original), n_samples)

    for idx in indices:
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        orig_pts = original[idx]
        ax1.plot(*orig_pts.T, 'b-o', markersize=4, alpha=0.6)
        ax1.set_title('Original Morphology')
        ax2 = fig.add_subplot(122, projection='3d')
        dist_pts = distorted[idx]
        ax2.plot(*dist_pts.T, 'r-o', markersize=4, alpha=0.6)
        ax2.set_title('Deformed Morphology')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    original_data = np.load(r'.\denosie\branches\pci_5000_true.npy')
    original_data = original_data[0:20000]
    print(original_data.shape)
    distorted_data = add_block_noise(original_data)

    assert np.allclose(original_data[:, 0], distorted_data[:, 0])
    assert np.allclose(original_data[:, -1], distorted_data[:, -1])
    # visualize_deformation(original_data, distorted_data)
    distorted_data2 = add_block_noise(original_data)
    distorted_data3 = add_block_noise(original_data)
    distorted_data = np.concatenate([distorted_data, distorted_data2, distorted_data3], axis=0)
    original_data = np.concatenate([original_data, original_data, original_data], axis=0)
    print(original_data.shape)
    print(distorted_data.shape)

    np.save(r'.\denosie\branches\pci_15000_true16.npy', original_data)
    np.save(r'.\denosie\branches\pci_15000_noise16.npy',distorted_data)