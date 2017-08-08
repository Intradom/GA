# This program uses a genetic algorithm to generate a hologram that matches an input image
# To run: python ../genetic_hologram_generator.py <path_to_hologram_templates> <path_to_OG_image_file>
# Image shape has to match template shape, JPEG images ONLY

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw as skdraw
from PIL import Image
from scipy import misc

# Image constants
MAX_INTENSITY = 255

# Parameters
END_THRESHOLD = 0.0000001 # RMSE cut-off, lower fitness value is better
MAX_SHAPE_SIZE = 50
ADDITIONS_BEFORE_EVAL = 10

def load_templates(templates_dir, templates):
    count = 0
    for file in os.listdir(templates_dir):
        if file.endswith(".jpg"):
            full_path = os.path.join(templates_dir, file)
            template_image = misc.imread(full_path, flatten = True)
            templates.append(template_image)
            count += 1
        
    return count

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
def fitness(original, current, image_side_len):
    # plt.imshow(current, cmap = 'gray', vmin = -255, vmax = 255) # Set vmin and vmax to force display not to automatically pick intensity range
    # plt.show()
    
    # Convert to grayscale
    #t_o = rgb2gray(original)
    #t_c = rgb2gray(current)
    #t = t_o - t_c
    
    t = original - current
    return np.linalg.norm(t) # Test out different norm functions

def draw_filled_circle(x, y, r):
    return skdraw.circle(int(x), int(y), int(r))
    
def draw_outline_circle(x, y, r):
    return skdraw.circle_perimeter(int(x), int(y), int(r))
    
def draw_filled_rectangle(x, y, half_side):
    return skdraw.polygon([x - half_side, x + half_side, x + half_side, x - half_side], [y - half_side, y - half_side, y + half_side, y + half_side])

def draw_outline_rectangle(x, y, half_side):
    return skdraw.polygon_perimeter([x - half_side, x + half_side, x + half_side, x - half_side], [y - half_side, y - half_side, y + half_side, y + half_side])

def add_hologram(holo_array, image_side_len, num_template_images, origin):
    # Create "blank canvas"        
    holo_vals = np.empty(6)

    # Determine which shape to add to canvas, and where to put it/size
    holo_vals[0] = np.random.randint(4) # 0 to 3
    holo_vals[1] = np.random.randint(MAX_SHAPE_SIZE, image_side_len - MAX_SHAPE_SIZE)
    holo_vals[2] = np.random.randint(MAX_SHAPE_SIZE, image_side_len - MAX_SHAPE_SIZE)
    holo_vals[3] = np.random.randint(1, MAX_SHAPE_SIZE) # Purposely allow for zero
    holo_vals[4] = 1 # Use as a mask // np.random.randint(-MAX_INTENSITY, MAX_INTENSITY + 1) # Max intensity is 255, random value from -255 to 255
    holo_vals[5] = np.random.randint(num_template_images)

    if (origin):
        holo_vals[1] = image_side_len / 2
        holo_vals[2] = image_side_len / 2
    
    holo_array.append(holo_vals)

def eval_fit(original_image, templates, holo_array, image_side_len, best_fitness):
    cumulative_hologram = np.zeros([image_side_len, image_side_len])
    for i in range(len(holo_array)):
        current_holo_vals = holo_array[i]
        mask = np.zeros([image_side_len, image_side_len])
        
        if (current_holo_vals[0] == 0):
            x, y = draw_filled_circle(current_holo_vals[1], current_holo_vals[2], current_holo_vals[3])
        elif (current_holo_vals[0] == 1):
            x, y = draw_outline_circle(current_holo_vals[1], current_holo_vals[2], current_holo_vals[3])
        elif (current_holo_vals[0] == 2):
            x, y = draw_filled_rectangle(current_holo_vals[1], current_holo_vals[2], current_holo_vals[3])
        elif (current_holo_vals[0] == 3):
            x, y = draw_outline_rectangle(current_holo_vals[1], current_holo_vals[2], current_holo_vals[3])

        mask[x, y] = current_holo_vals[4]
        temp_holo = np.multiply(mask, templates[int(current_holo_vals[5])])
        
        cumulative_hologram += temp_holo
        
        # plt.imshow(cumulative_hologram, cmap = 'gray')
    
    # TESTING, show image of current cumulative hologram
    # plt.imshow(np.fft.fftshift(cumulative_hologram), cmap = 'gray', vmin = -255, vmax = 255) # Set vmin and vmax to force display not to automatically pick intensity range
    # plt.show()

    holo_revert = np.fft.fftshift(np.fft.ifft2(cumulative_hologram)[:image_side_len][:image_side_len].real)
    
    # Evaluate fitness of current cumulative hologram
    fit = fitness(original_image, holo_revert, image_side_len)
    
    if (fit < best_fitness):
        best_fitness = fit
        print("New best fitness: ", fit)
        plt.figure(1)
        plt.imshow(cumulative_hologram, cmap = 'gray')
        plt.figure(2)
        plt.imshow(holo_revert, cmap = 'gray')
        plt.show()
    else:
        holo_array = holo_array[:len(holo_array) - ADDITIONS_BEFORE_EVAL] # Remove the recently added-in hologram images
            
    return best_fitness

def main():
    # Load templates
    templates_dir = str(sys.argv[1])
    templates = []
    NUM_TEMPLATES = load_templates(templates_dir, templates)
    templates = np.array(templates)
        
    # Load target image
    pic_path = str(sys.argv[2])
    OG_image = misc.imread(pic_path, flatten=True)
    plt.imshow(OG_image, cmap = 'gray')
    plt.show()
    image_side_len = OG_image.shape[0] # Shape has to be same size as template shape
    
    # Create Hologram array to store individual shapes
    holo_array = []

    best_fitness = sys.float_info.max # Lower fitness is better, initialize high
    iter_counter = 0
    # Loop until fitness eval is within error threshold
    while (best_fitness > END_THRESHOLD):    
        for i in range(ADDITIONS_BEFORE_EVAL):
            if (i == 0):
                add_hologram(holo_array, image_side_len, NUM_TEMPLATES, True) # holo_array is modified in the function
            else:
                add_hologram(holo_array, image_side_len, NUM_TEMPLATES, False) # holo_array is modified in the function
                
        best_fitness = eval_fit(OG_image, templates, holo_array, image_side_len, best_fitness)
        
        iter_counter += 1
    
    # Display and save final hologram and its corresponding reconnstructed image

if __name__ == "__main__":
    main()
