import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Define the dimensions of the area
x_min, x_max = 0, 20
y_min, y_max = 0, 20

# Generate 50 random points
x_coords = np.random.uniform(x_min, x_max, 50)
y_coords = np.random.uniform(y_min, y_max, 50)
points = np.column_stack((x_coords, y_coords))

# Random starting point
start_idx = np.random.randint(0, len(points))
current_point = points[start_idx]
visited = [start_idx]

# Traverse using the greedy algorithm (Nearest Neighbor)
while len(visited) < len(points):
    # Find the indices of remaining unvisited points
    remaining_points = [i for i in range(len(points)) if i not in visited]
    
    # Calculate distances from the current point to all unvisited points
    distances = cdist([current_point], points[remaining_points])[0]
    
    # Find the nearest unvisited point
    nearest_idx = remaining_points[np.argmin(distances)]
    
    # Mark it as visited and update the current point
    visited.append(nearest_idx)
    current_point = points[nearest_idx]

# Add the starting point to complete the cycle (optional for visualization)
visited.append(visited[0])

# Calculate total traversal distance
traversal = points[visited]
total_distance = np.sum(np.linalg.norm(traversal[:-1] - traversal[1:], axis=1))

# Time to traverse (time = distance / speed)
speed = 1.2  # units per second
total_time = total_distance / speed

# Display the traversal time
print(f"Total traversal distance: {total_distance:.2f} units")
print(f"Total traversal time: {total_time:.2f} seconds")

# Plot the results
plt.figure(figsize=(8, 8))

# Scatter plot of all points
plt.scatter(x_coords, y_coords, color='blue', label='Points')
for i, (x, y) in enumerate(points):
    plt.text(x, y, str(i), fontsize=8, ha='right')  # Annotate points with indices

# Plot the traversal path
plt.plot(traversal[:, 0], traversal[:, 1], color='red', label='Path', linewidth=1.5)

# Add labels and title
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title("Randomly Scattered Balls After Multiball Session")
plt.xlabel("length of training grounds")
plt.ylabel("breadth of training grounds")
plt.legend()
plt.grid(True)
plt.show()

