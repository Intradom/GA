# This program uses a genetic algorithm to generate a hologram that matches an input image

import numpy as np
import matplotlib.pyplot as plt
import skimage.draw as skdraw

# Square pictures only
IMAGE_SIDE_PIXELS = 480

# Parameters
MAX_SHAPE_SIZE = 50

def evaluate_fitness(hologram):
    recon_image = np.fft.ifft2(np.fft.fftshift(hologram))
    plt.imshow(recon_image.real, cmap = 'gray')
    plt.show()
    
    return 0

def draw_filled_circle(x, y, r):
    return skdraw.circle(int(x), int(y), int(r))
    
def draw_outline_circle(x, y, r):
    return skdraw.circle_perimeter(int(x), int(y), int(r))
    
def draw_filled_rectangle(x, y, half_side):
    return skdraw.polygon([x - half_side, x + half_side, x + half_side, x - half_side], [y - half_side, y - half_side, y + half_side, y + half_side])

def draw_outline_rectangle(x, y, half_side):
    return skdraw.polygon_perimeter([x - half_side, x + half_side, x + half_side, x - half_side], [y - half_side, y - half_side, y + half_side, y + half_side])

def main():
    # Load target image
    
    # Create Hologram array to store individual shapes
    holo_array = []

    iter_counter = 0
    # Loop until fitness eval is within error threshold
    while (True):
        print("Iteration: %d"  % iter_counter)
    
        # Create "blank canvas"        
        holo_vals = np.empty(5)
        
        # Determine which shape to add to canvas, and where to put it/size
        holo_vals[0] = np.random.randint(4) # 0 to 3
        holo_vals[1] = np.random.randint(MAX_SHAPE_SIZE, IMAGE_SIDE_PIXELS - MAX_SHAPE_SIZE)
        holo_vals[2] = np.random.randint(MAX_SHAPE_SIZE, IMAGE_SIDE_PIXELS - MAX_SHAPE_SIZE)
        holo_vals[3] = np.random.randint(MAX_SHAPE_SIZE) # Purposely allow for zero
        holo_vals[4] = np.random.randint(-255, 256) # Max intensity is 255, random value from -255 to 255
                
        holo_array.append(holo_vals)
                
        if (np.mod(iter_counter, 100) == 0):
            cumulative_hologram = np.zeros([IMAGE_SIDE_PIXELS, IMAGE_SIDE_PIXELS])
            for i in range(len(holo_array)):
                
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
            plt.imshow(cumulative_hologram, cmap = 'gray', vmin = 0, vmax = 255) # Set vmin and vmax to force display not to automatically pick intensity range
            plt.show()
        
            # Evaluate fitness of current cumulative hologram
            evaluate_fitness(cumulative_hologram)
        
        iter_counter += 1
    
    # Display and save final hologram and its corresponding reconnstructed image

if __name__ == "__main__":
    main()