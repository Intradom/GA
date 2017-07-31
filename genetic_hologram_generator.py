# This program uses a genetic algorithm to generate a hologram that matches an input image

import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw as skdraw

# Square pictures only
IMAGE_SIDE_PIXELS = 480

# Image constants
MAX_INTENSITY = 255

# Parameters
END_THRESHOLD = 0.1 # Percentage difference purely in difference of pixel intensity relative to max difference, lower fitness value is better
MAX_SHAPE_SIZE = 50
ADDITIONS_BEFORE_EVAL = 10

def fitness(original, current):
    # plt.imshow(current, cmap = 'gray', vmin = -255, vmax = 255) # Set vmin and vmax to force display not to automatically pick intensity range
    # plt.show()

    t = original - current
    return np.sum(t) / (MAX_INTENSITY * 2 * np.square(IMAGE_SIDE_PIXELS))

def draw_filled_circle(x, y, r):
    return skdraw.circle(int(x), int(y), int(r))
    
def draw_outline_circle(x, y, r):
    return skdraw.circle_perimeter(int(x), int(y), int(r))
    
def draw_filled_rectangle(x, y, half_side):
    return skdraw.polygon([x - half_side, x + half_side, x + half_side, x - half_side], [y - half_side, y - half_side, y + half_side, y + half_side])

def draw_outline_rectangle(x, y, half_side):
    return skdraw.polygon_perimeter([x - half_side, x + half_side, x + half_side, x - half_side], [y - half_side, y - half_side, y + half_side, y + half_side])

def add_hologram(holo_array):
    # Create "blank canvas"        
    holo_vals = np.empty(5)

    # Determine which shape to add to canvas, and where to put it/size
    holo_vals[0] = np.random.randint(4) # 0 to 3
    holo_vals[1] = np.random.randint(MAX_SHAPE_SIZE, IMAGE_SIDE_PIXELS - MAX_SHAPE_SIZE)
    holo_vals[2] = np.random.randint(MAX_SHAPE_SIZE, IMAGE_SIDE_PIXELS - MAX_SHAPE_SIZE)
    holo_vals[3] = np.random.randint(MAX_SHAPE_SIZE) # Purposely allow for zero
    holo_vals[4] = np.random.randint(-MAX_INTENSITY, MAX_INTENSITY + 1) # Max intensity is 255, random value from -255 to 255

    holo_array.append(holo_vals)

def eval_fit(original_image, holo_array)
    cumulative_hologram = np.zeros([IMAGE_SIDE_PIXELS, IMAGE_SIDE_PIXELS])
    for (i in range(len(holo_array))):
        
        current_holo_vals = holo_array[i]
        temp_holo = np.zeros([IMAGE_SIDE_PIXELS, IMAGE_SIDE_PIXELS])
        
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
    return fitness(original_image, np.fft.ifft2(cumulative_hologram)[:IMAGE_SIDE_PIXELS][:IMAGE_SIDE_PIXELS].real)

def main():
    # Load target image
    
    # Create Hologram array to store individual shapes
    holo_array = []

    best_fitness = (MAX_INTENSITY * 2 * np.square(IMAGE_SIDE_PIXELS)) # Max value fitness can take
    iter_counter = 0
    # Loop until fitness eval is within error threshold
    while (best_fitness > END_THRESHOLD):
        print("Iteration: %d"  % iter_counter)
    
        for (i in range(ADDITIONS_BEFORE_EVAL)):
            add_hologram(holo_array) # holo_array is modified in the function
                
        current_fitness = eval_fit(holo_array)
        
        if (current_fitness < best_fitness):
            best_fitness = current_fitness
        else
            holo_array = holo_array[:len(holo_array) - ADDITIONS_BEFORE_EVAL] # Remove the recently added-in hologram images
        
        iter_counter += 1
    
    # Display and save final hologram and its corresponding reconnstructed image

if __name__ == "__main__":
    main()
