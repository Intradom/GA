# This program uses a genetic algorithm to generate a hologram that matches an input image
# To run: python ../genetic_hologram_generator.py <path_to_OG_image_file>

import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw as skdraw
from scipy import misc

# Image constants
MAX_INTENSITY = 255

# Parameters
END_THRESHOLD = 0.01 # Percentage difference purely in difference of pixel intensity relative to max difference, lower fitness value is better
MAX_SHAPE_SIZE = 50
ADDITIONS_BEFORE_EVAL = 10

def fitness(original, current, image_side_len):
    # plt.imshow(current, cmap = 'gray', vmin = -255, vmax = 255) # Set vmin and vmax to force display not to automatically pick intensity range
    # plt.show()

    t = original - current
    return np.sum(t) / (MAX_INTENSITY * 2 * np.square(image_side_len))

def draw_filled_circle(x, y, r):
    return skdraw.circle(int(x), int(y), int(r))
    
def draw_outline_circle(x, y, r):
    return skdraw.circle_perimeter(int(x), int(y), int(r))
    
def draw_filled_rectangle(x, y, half_side):
    return skdraw.polygon([x - half_side, x + half_side, x + half_side, x - half_side], [y - half_side, y - half_side, y + half_side, y + half_side])

def draw_outline_rectangle(x, y, half_side):
    return skdraw.polygon_perimeter([x - half_side, x + half_side, x + half_side, x - half_side], [y - half_side, y - half_side, y + half_side, y + half_side])

def add_hologram(holo_array, image_side_len):
    # Create "blank canvas"        
    holo_vals = np.empty(5)

    # Determine which shape to add to canvas, and where to put it/size
    holo_vals[0] = np.random.randint(4) # 0 to 3
    holo_vals[1] = np.random.randint(MAX_SHAPE_SIZE, image_side_len - MAX_SHAPE_SIZE)
    holo_vals[2] = np.random.randint(MAX_SHAPE_SIZE, image_side_len - MAX_SHAPE_SIZE)
    holo_vals[3] = np.random.randint(MAX_SHAPE_SIZE) # Purposely allow for zero
    holo_vals[4] = np.random.randint(-MAX_INTENSITY, MAX_INTENSITY + 1) # Max intensity is 255, random value from -255 to 255

    holo_array.append(holo_vals)

def eval_fit(original_image, holo_array, image_side_len):
    cumulative_hologram = np.zeros([image_side_len, image_side_len])
    for i in range(len(holo_array)):
        current_holo_vals = holo_array[i]
        temp_holo = np.zeros([image_side_len, image_side_len])
        
        if (current_holo_vals[0] == 0):
            x, y = draw_filled_circle(current_holo_vals[1], current_holo_vals[2], current_holo_vals[3])
        elif (current_holo_vals[0] == 1):
            x, y = draw_outline_circle(current_holo_vals[1], current_holo_vals[2], current_holo_vals[3])
        elif (current_holo_vals[0] == 2):
            x, y = draw_filled_rectangle(current_holo_vals[1], current_holo_vals[2], current_holo_vals[3])
        elif (current_holo_vals[0] == 3):
            x, y = draw_outline_rectangle(current_holo_vals[1], current_holo_vals[2], current_holo_vals[3])

        temp_holo[x, y] = current_holo_vals[4]
        
        cumulative_hologram += temp_holo
    
    # TESTING, show image of current cumulative hologram
    # plt.imshow(np.fft.fftshift(cumulative_hologram), cmap = 'gray', vmin = -255, vmax = 255) # Set vmin and vmax to force display not to automatically pick intensity range
    # plt.show()

    # Evaluate fitness of current cumulative hologram
    return fitness(original_image, np.fft.ifft2(cumulative_hologram)[:image_side_len][:image_side_len].real, image_side_len)

def main():
    # Load target image
    pic_name = str(sys.argv[1])
    OG_image = misc.imread(pic_name, flatten=True)
    image_side_len = OG_image.shape[0]
    
    # Create Hologram array to store individual shapes
    holo_array = []

    best_fitness = (MAX_INTENSITY * 2 * np.square(image_side_len)) # Max value fitness can take
    iter_counter = 0
    # Loop until fitness eval is within error threshold
    while (best_fitness > END_THRESHOLD):    
        for i in range(ADDITIONS_BEFORE_EVAL):
            add_hologram(holo_array, image_side_len) # holo_array is modified in the function
                
        current_fitness = eval_fit(OG_image, holo_array, image_side_len)
        
        if (current_fitness < best_fitness):
            best_fitness = current_fitness
            print("New best fitness: ", current_fitness)
        else:
            holo_array = holo_array[:len(holo_array) - ADDITIONS_BEFORE_EVAL] # Remove the recently added-in hologram images
        
        iter_counter += 1
    
    # Display and save final hologram and its corresponding reconnstructed image

if __name__ == "__main__":
    main()
