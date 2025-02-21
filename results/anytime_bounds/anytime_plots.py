import matplotlib.pyplot as plt
import csv

# Read data from CSV file
def read_csv(filename):
    times, lower_bounds, upper_bounds = [], [], []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            times.append(float(row[0]))
            lower_bounds.append(float(row[1]))
            upper_bounds.append(float(row[2]))
    return times, lower_bounds, upper_bounds

# Define file path
csv_file = 'results/anytime_bounds/helicopter_12_versus_all_anytime_bounds.csv'  # Change this to your actual file path
save_path = 'plots/' +'cartpole' +  '.png'

# Read data
# times, lower_bounds, upper_bounds = read_csv(csv_file)

# Extra points with only one bound value
extra_points = [
    (21.696, 2, 'dtControl'),
    (0.13838434219360352, 2, 'StreeD'),
    (.865, 2, 'Witty')
]

# Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(times, lower_bounds, label='Lower Bound', marker='o', linestyle='-', color='red')
# plt.plot(times, upper_bounds, label='Upper Bound', marker='s', linestyle='-', color='blue')

# Plot extra points
for t, val, label in extra_points:
    plt.scatter(t, val, label=label, marker='x', s=100)

# Labels and legend
plt.xlabel('Time (seconds)')
plt.ylabel('Bounds')
plt.title('Anytime Algorithm Bounds')
plt.legend()
plt.grid(True)

# Save and show the plot
plt.savefig(save_path)
plt.show()
