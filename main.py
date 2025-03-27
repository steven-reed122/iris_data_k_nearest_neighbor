import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

def get_iris_data_list():
    iris_data = []
    with open('iris.csv', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            iris_data.append([float(row[0]), float(row[1]), row[4]])  # Sepal length, Sepal width, Variety
    return iris_data

# Load and prepare data
iris_data = get_iris_data_list()
x = np.array([[point[0], point[1]] for point in iris_data])  # Features
y = np.array([point[2] for point in iris_data])  # Labels

# Map class labels to integers
class_mapping = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
y = np.array([class_mapping[label] for label in y])

# Train k-NN classifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x, y)

# Define decision boundary grid
x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Set tick intervals to 0.5
plt.xticks(np.arange(4.5, 8.0, 0.5))  # X-axis ticks every 0.5
plt.yticks(np.arange(2.0, 4.5, 0.5))  # Y-axis ticks every 0.5

# Plot decision boundary
colors = ['purple', 'teal', 'yellow']
cmap = ListedColormap(colors)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

# Scatter plot of data points
for variety, color in class_mapping.items():
    plt.scatter(x[y == color, 0], x[y == color, 1], c=colors[color], label=variety, edgecolors='k')


plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('k-NN Decision Boundary (k=7)')
plt.legend()
plt.show()
