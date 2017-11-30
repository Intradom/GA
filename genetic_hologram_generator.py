# This program uses a genetic algorithm to generate a hologram that matches an input image
# To run: python ../genetic_hologram_generator.py <path_to_hologram_templates> <path_to_OG_image_file>
# Image shape has to match template shape, JPEG images ONLY

import os
import sys
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw as skdraw
from PIL import Image
from scipy import misc

# Image constants
MAX_INTENSITY = 255

# Parameters
OG_IMAGE_DIMMING_FACTOR = 10000 # So reconstructed values can come close to OG image's values, otherwise OG image too bright for iFFT to match quickly. Target value = 255 / OG_IMAGE_DIMMING_FACTOR
END_THRESHOLD = 0.0000001 # RMSE cut-off, lower fitness value is better. IGNORE THIS VALUE RIGHT NOW, not properly tuned
END_SGENS = 200
ADDITIONS_BEFORE_EVAL = 10
SUCCESSFUL_GENS_BEFORE_OUTPUT = 50
NUM_INITIAL_LINES = 2 # How many initial parallel holograms to evolve. More than 1 enables cross-breeding.
MAX_LINES = 5
CROSSOVER_RATE = 50 # Number of S_gens before crossover
CROSSOVER_SINGLE_POINT_THRESH = 0.5 # Value between 0 and 1 that determines where the code is swapped at. Higher values means more of original chromosome is preserved

# Parameters to loop through, add elements to these arrays to loop through them
MAX_SHAPE_SIZE = [10]
MUTATION_CHANCE = [0.1] # Value between 0 and 1, 0 means no mutations while 1 means all mutations and no new layers, should be at most 2 decimal precsion value for file directory saving

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
    #np.set_printoptions(threshold=np.nan) # Set print to infinite values
    #np.savetxt('original_image.txt', original)
    #np.savetxt('current_image.txt', current)
    #print(np.amax(current))
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

def add_hologram(holo_array, image_side_len, num_template_images, origin, s):
    # Create "blank canvas"        
    holo_vals = np.empty(6)

    # Determine which shape to add to canvas, and where to put it/size
    holo_vals[0] = np.random.randint(4) # 0 to 3
    holo_vals[1] = np.random.randint(s, image_side_len - s)
    holo_vals[2] = np.random.randint(s, image_side_len - s)
    holo_vals[3] = np.random.randint(1, s) # Purposely allow for zero
    holo_vals[4] = np.random.random() * 2 - 1 # Use as a pos mask or neg partial mask // np.random.randint(-MAX_INTENSITY, MAX_INTENSITY + 1) # Max intensity is 255, random value from -255 to 255
    holo_vals[5] = np.random.randint(num_template_images)

    if (origin):
        holo_vals[1] = image_side_len / 2
        holo_vals[2] = image_side_len / 2
    
    holo_array.append(holo_vals)

""" Crossover Functions """
def crossover_helper_targ_random(current_line, total_lines):
    # Select another line besides current one
    r_array = np.asarray(range(total_lines))
    np.random.shuffle(r_array)
    targ_line = r_array[0]
    if (targ_line == current_line):
        targ_line = r_array[1]
        
    return targ_line

def crossover_two_lines_single_point(new_line_array, holo_arrays, current_line, total_lines):
    targ_line = crossover_helper_targ_random(current_line, total_lines)
    
    curr_use_len = int(len(holo_arrays[current_line]) * CROSSOVER_SINGLE_POINT_THRESH)
    targ_use_len = int(len(holo_arrays[targ_line]) * CROSSOVER_SINGLE_POINT_THRESH)
    
    new_line_array = copy.deepcopy(holo_arrays[current_line][:curr_use_len]) # Before thresh
    new_holo_after_thresh = copy.deepcopy(holo_arrays[targ_line][targ_use_len:]) # After thresh
    
    print("Crossed line " + str(current_line) + " with line " + str(targ_line))
    
    new_line_array.append(new_holo_after_thresh)
    
def eval_fit(original_image, templates, holo_array, image_side_len, best_fitness, current_sgen, start_time, m, s, lin_num):
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

    holo_revert = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(cumulative_hologram))[:image_side_len][:image_side_len].real)
    
    # Evaluate fitness of current cumulative hologram
    fit = fitness(original_image, holo_revert, image_side_len)
    
    best_fit = False
    if (fit < best_fitness):
        best_fitness = fit
        print("Line " + str(lin_num) + " new best fitness: %.5f -- S_gen %d" % (fit, current_sgen + 1))
        # Don't output so many times
        if ((current_sgen + 1) % SUCCESSFUL_GENS_BEFORE_OUTPUT == 0 or current_sgen == 0):
            # Make sure directory exists
            target_dir = "Outputs/Current_run/" + "Mutate_" + str(int(m * 100)) + "/Max_shape_" + str(s) + "/"
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)
        
            #plt.figure(1)
            plt.imshow(cumulative_hologram, cmap = 'gray')
            plt.title("Hologram Line " + str(lin_num))
            plt.savefig(target_dir + "S_gen_" + str(current_sgen + 1) + "_holo" + "_Line" + str(lin_num) + ".png")
            #plt.figure(2)
            et_h, rem = divmod(int(time.time() - start_time), 3600)
            et_m, et_s = divmod(rem, 60)
            plt.imshow(holo_revert, cmap = 'gray')
            plt.title("Reverted Line " + str(lin_num) + "- fit: " + "{0:.5f}".format(fit) + ", elapsed time: " + str(et_h) + "h " + str(et_m) + "m " + str(et_s) + "s")
            plt.savefig(target_dir + "S_gen_" + str(current_sgen + 1) + "_revert" + "_Line" + str(lin_num) + ".png")
            #plt.show()
        current_sgen += 1
        best_fit = True
        
    # return best_fitness, current_sgen, holo_array[:(len(holo_array) - ADDITIONS_BEFORE_EVAL)] # Remove the recently added-in hologram images
    return best_fitness, current_sgen, best_fit

def main():
    # Load templates
    print("Loading templates")
    templates_dir = str(sys.argv[1])
    templates = []
    NUM_TEMPLATES = load_templates(templates_dir, templates)
    templates = np.array(templates)
        
    # Load target image
    print("Loading target image")
    pic_path = str(sys.argv[2])
    OG_image = misc.imread(pic_path, flatten=True)
    OG_image /= OG_IMAGE_DIMMING_FACTOR # Dimming so GA can reach optimal brightness faster
    # Uncomment to show template image at beginning
    # plt.imshow(OG_image, cmap = 'gray')
    # plt.show()
    image_side_len = OG_image.shape[0] # Shape has to be same size as template shape
    
    for m in range(len(MUTATION_CHANCE)):
        for s in range(len(MAX_SHAPE_SIZE)):
            print("Starting run of Mutate: " + str(MUTATION_CHANCE[m]) + " and Max shape size: " + str(MAX_SHAPE_SIZE[s]))
            # Create Hologram array to store individual shapes
            holo_arrays = []
            lines_completed = 0
            
            # Get initial time
            start_time = time.time()
            
            best_fitness = []
            line_array = []
            line_finished = []
            current_sgen = []
            can_cross = []
            for i in range(NUM_INITIAL_LINES):
                holo_arrays.append(line_array)
                best_fitness.append(sys.float_info.max) # Lower fitness is better, initialize high
                line_finished.append(False)
                can_cross.append(False)
                current_sgen.append(0)
            num_total_lines = NUM_INITIAL_LINES
                
            # Loop until fitness eval is within error threshold
            while (lines_completed < num_total_lines):
                for line in range(num_total_lines):
                    if (not line_finished[line]):
                        if (best_fitness[line] > END_THRESHOLD and current_sgen[line] < END_SGENS):
                            if ((current_sgen[line] % CROSSOVER_RATE == 0) and (can_cross[line]) and (num_total_lines < MAX_LINES) and (num_total_lines > 1)):
                                # Create new line by crossbreeding existing lines_completed
                                new_line_array = []
                                
                                crossover_two_lines_single_point(new_line_array, holo_arrays, line, num_total_lines) # Modifies new_line_array
                                holo_arrays.append(new_line_array)
                                best_fitness.append(sys.float_info.max)
                                line_finished.append(False)
                                can_cross.append(False)
                                current_sgen.append(current_sgen[line]) # Keep new line at same S_gen mark so it's not forced into s_gen asymptote too early
                                num_total_lines += 1
                                
                                can_cross[line] = False
                                
                            tmp_holo_array = copy.deepcopy(holo_arrays[line])
                            for i in range(ADDITIONS_BEFORE_EVAL):
                                r = np.random.random() # Generates a random value between 0 and 1
                                if (r < m and len(tmp_holo_array) > 0): # Mutate, remove hologram layer because layers are additive
                                    r_int = np.random.randint(len(tmp_holo_array))
                                    del tmp_holo_array[r_int]
                                # Augment  
                                if (i == 0):
                                    add_hologram(tmp_holo_array, image_side_len, NUM_TEMPLATES, True, MAX_SHAPE_SIZE[s]) # tmp_holo_array is modified in the function, includes point in origin
                                else:
                                    add_hologram(tmp_holo_array, image_side_len, NUM_TEMPLATES, False, MAX_SHAPE_SIZE[s]) # tmp_holo_array is modified in the function             
                            
                            best_fitness[line], current_sgen[line], best_fit = eval_fit(OG_image, templates, tmp_holo_array, image_side_len, best_fitness[line], current_sgen[line], start_time, MUTATION_CHANCE[m], MAX_SHAPE_SIZE[s], line)
                            
                            if (best_fit):
                                holo_arrays[line] = tmp_holo_array
                                can_cross[line] = True
                        else:
                            line_finished[line] = True
                            lines_completed += 1
                        
if __name__ == "__main__":
    main()
