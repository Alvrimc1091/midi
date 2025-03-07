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
scale_factor = 1/3.0

# Get image dimensions
height, width = image.shape[:2]

# Calculate the new image dimensions
new_height = int(height * scale_factor)
new_width = int(width * scale_factor)

# Create a black mask
mask = np.zeros((height, width), dtype=np.uint8)

# Define circle center and radius
# center = (width // 2, int(height * 0.9))  # Moves the circle higher
#  # Center of the image
# radius = min(width, height) // 3  # Fit within the image

# Define non-integer center and radius
center_x = width / 2.02  # Example: keep it centered
center_y = height * 0.39 # Move the circle up slightly
radius = min(width, height) / 3.8  # Adjust the radius proportionally

# Convert to integers for OpenCV
center = (int(center_x), int(center_y))
radius = int(radius)

# Draw a filled white circle
cv2.circle(mask, center, radius, 255, thickness=-1)

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
