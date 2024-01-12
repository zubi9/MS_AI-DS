import numpy as np
import matplotlib.pyplot as plt

# Fractal dimension calculation using the box-counting method
def box_counting_dimension(data, num_boxes_per_side=10):
    min_x, max_x = np.min(data[:, 0]), np.max(data[:, 0])
    min_y, max_y = np.min(data[:, 1]), np.max(data[:, 1])

    box_size_x = (max_x - min_x) / num_boxes_per_side
    box_size_y = (max_y - min_y) / num_boxes_per_side

    box_counts = np.zeros((num_boxes_per_side, num_boxes_per_side))

    for point in data:
        box_x = int((point[0] - min_x) / box_size_x)
        box_y = int((point[1] - min_y) / box_size_y)
        if box_x < num_boxes_per_side and box_y < num_boxes_per_side:
            box_counts[box_x, box_y] = 1

    non_empty_boxes = np.sum(box_counts)
    return np.log(non_empty_boxes) / np.log(1.0 / min(box_size_x, box_size_y))

# Generate synthetic fractal-like data (Koch Snowflake)
def koch_snowflake(points, iterations):
    result = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        delta = (p2 - p1) / 3.0
        result.extend([
            p1,
            p1 + delta,
            p1 + delta + delta.imag * 1j,
            p1 + delta + delta.imag * 1j + delta * 1j,
        ])
    result.append(points[-1])
    if iterations == 1:
        return result
    else:
        return koch_snowflake(result, iterations - 1)

# Starting points of an equilateral triangle
triangle_points = np.array([
    [0, 0],
    [0.5, np.sqrt(3) / 2],
    [1, 0],
    [0, 0]
])

# Generate Koch Snowflake data
snowflake_data = np.array(koch_snowflake(triangle_points, iterations=10))

# Calculate box-counting fractal dimension
box_count_dimension = box_counting_dimension(snowflake_data)
print("Box-Counting Fractal Dimension (Koch Snowflake):", box_count_dimension)

# Plot the Koch Snowflake pattern
plt.plot(snowflake_data[:, 0], snowflake_data[:, 1])
plt.title("Fractal-Like Pattern (Koch Snowflake)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
