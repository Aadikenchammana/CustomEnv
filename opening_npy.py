import numpy as np
import matplotlib.pyplot as plt

def create_image_from_npy(npy_file):
    # Load data from npy file
    data = np.load(npy_file)

    # Create an image based on the data
    plt.imshow(data, cmap='viridis')  # You can change the cmap to suit your data
    plt.colorbar()  # Add a colorbar for reference
    plt.show()

# Replace 'your_file.npy' with the actual file path
for i in range(100):
    npy_file_path = 'train_analytics//02.09.2024_18:15:02//fires//0//'+str(i+1)+'.npy'
    create_image_from_npy(npy_file_path)
