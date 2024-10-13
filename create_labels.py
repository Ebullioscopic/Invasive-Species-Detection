import numpy as np

# Create a 10x10 sample of ground truth labels (1 for Kudzu, 0 for non-Kudzu)
ground_truth_labels = np.array([
    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 1, 1, 0, 0, 1, 0]
])

# Save the ground truth labels as a .npy file
np.save('ground_truth_labels.npy', ground_truth_labels)

print("Ground truth labels saved to ground_truth_labels.npy")
