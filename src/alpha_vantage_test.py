from utils import y_numeric_to_vector
import numpy as np
data = np.array([0, 1, 2, 3, 1, 1, 1, 2, 0])
print(y_numeric_to_vector(data, 4))