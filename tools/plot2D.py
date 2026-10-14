import matplotlib.pyplot as plt

# Sample data
points = [
  (-22.84, 25), # 1
  (-17.89, 25), # 2
  (-12.94, 25), # 3
  (-7.98, 25),  # 4
  (-3.02, 25),  # 5
  (1.94, 25),   # 6
  (5, 30),      # 7
  (-26, 30),    # 8
  (-26, 8.5),   # 9
  (-26, -23),   # 10
  (5, -23),     # 11
  (5, 8.5),     # 12
]

fig, ax = plt.subplots()

# Create the scatter plot
x, y = zip(*points)
ax.scatter(x, y, color="blue")

# Loop through the data points to add labels
for i, point in enumerate(points):
  ax.annotate(
      str(i + 1),
      point,  # Position of the point
      xytext=(5, 5),  # Distance in points from the point (x, y)
      textcoords="offset points",  # How to interpret xytext
      fontsize=10,
  )

plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.show()