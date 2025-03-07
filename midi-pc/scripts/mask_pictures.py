import cv2 # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend

# Load image
image = cv2.imread('/home/alvaro/Documents/Workspace/midi/data/static_training/bicarbonato/20250111_153943_foto.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define the scale factor 
# Decrease the size by 3 times
scale_factor = 1/2.0

# Get image dimensions
height, width = image.shape[:2]

# Calculate the new image dimensions
new_height = int(height * scale_factor)
new_width = int(width * scale_factor)

# Create a black mask
mask = np.zeros((height, width), dtype=np.uint8)

# Define circle parameters (adjust as needed)
circle_x = width // 2  # Adjustable horizontal position
circle_y = int(height * 0.38)  # Adjustable vertical position
circle_radius = min(width, height) // 3 # Adjustable size

# Draw a filled white circle (complete and adjustable)
cv2.circle(mask, (circle_x, circle_y), circle_radius, 255, thickness=-1)

# Apply mask to the image
masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

print(masked_image.shape)

# Scaled image
scaled_image = cv2.resize(src= masked_image, 
                          dsize =(new_width, new_height), 
                          interpolation=cv2.INTER_AREA)

print(scaled_image.shape)

# Show results
plt.plot()
plt.imshow(scaled_image)
#plt.title("3/4 Circle Masked Image")

plt.axis('off')

plt.show()

plt.savefig("scaled_image_grain2.png")
