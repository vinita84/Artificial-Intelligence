#!/usr/bin/env python

#------------> NNET description------------------>
'''I have implemented image orientation classification using the following architecture of neural networks:
hidden layers:
bias:1
learning rate:
input size: 192
number of output classes = 4
1) I formulated the problems as : the image data would act as my input ,an array of shape (192,1)
Output vector would be (4,1).
These 192 inputs will pass through the above described neural network in the following sequence:
A. forward propogation will predict the output of these input neuron using randomly generated weights
B. Softmax layer will calculate the error
C. Back propogation will propogate the errors till the input layer
D. Backpropogation will update the weights at each step
E. The steps from A through D will take place for all images in one epoch
F. the number of epochs will be decided by the drop in mean error. If mean error stagnates, the training will stop
assuming a minima has been reached.

2)WORKING: this program works by passing the following parameters:
    ./orient01.py train train_file.txt model_file.txt [model]
    it checks if the action to be performed is 'train', it will do the training else if it is 'test' it will do the testing.
    for each of these actions it will use the mentioned train or test file name for reading data, model filename for
    reading parameters and the model name provided for using the model

3) In design decisions, I have used stochastic gradient descent and sigmoid as activation function. These were chosen as
SGD improves the accuracy and sigmoid was one of thepopular activation functions.
I also tried alpha decay so that the algorithm converges to the global minima.

ASSUMPTIONS / CONSTRAINTS
***NOTE: if a model file with parameters is provided at the time of training, this neural net will not take those parameters as
initial weights when it starts for the first time. The initial weights are random.

'''

#------------> KNN description------------------>

#!/usr/bin/python2
# K nearest neighbours algorithm:
# Here, the RGB space is used to get Euclidiean "distances" between pixels.
# The distance between each pixel is the square root of ((r1-r1)^2 + (g1-g2)^2 + (b1-b2)^2)
# where r1,g1,b1 and r2,g2,b2 are the RGB pixel values for each image.
# The sum of all such distances for every corresponding pixel in two images is the effective "distance"
# between the two images.
# Normalizing (dividing by 255) and converting to HSV (hue, saturation, value) space
# did not improve the accuracy much, hence the RGB pixel values were fed as-is to the algorithm.
# The distances are sorted and the closest 'k' neighbours vote for the correct orientation.
# The orientation that gets maximum number of votes from its nearest neighbours is assigned to the image.
# The model file "nearest_model.txt" contains the k-value, which may be modified for testing purposes.
# Getting Euclidean distance directly from numpy arrays :
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html#numpy.linalg.norm
# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy


############## Adaboost ############3

# Set of states:
# All possible stumps (192 * 192)
#
# Initial State:
# Empty array of stumps
#
#
# Goal:
# Set of k stumps for which
# all examples x get correct label
# h(stump)(x) = correct label
#
# Successor:
# for k (0,K) do:
# 	for all examples, find stump with minimum error (best stump)
# 	update weights for each of the examples
#
# Error:
# incorrect = 0
# all examples x in test
# 	get correct label
# 	get weighted_majority_of_all_k_stumps for all classifiers (0,90,180,270)
# 	find predicted = max(weighted_majority_of_all_k_stumps for all classifiers 0,90,180,270)
# 	predicted != actual
# 		incorrect += 1
# Error: (incorrect/total)

# Brief Description:
# The main idea behind using boosting is to have a set of weak classifiers, each with a confidence value, such that they together give us the best possible prediction ie. correct orientation label for a all the examples in test.
#
# Eventually, the error rate converges to 0 for examples in train file and
# the error for examples in test file becomes stagnant
#
# In other words, ideally the accuracy of the weighted classifier becomes almost 100% for train examples
# and accuracy becomes constant for test examples (say ~70%)
#
# The algorithm implemented below for adaboost, is by Freund & Schapire 1995
# We have implemented 4 classifiers, one for each (0,180,90,270)

# For each classifier:
# 	selected_stumps = list()
#     z_alpha_values = list()
# 	num_of_positive_exemplars = 9244
# 	num_of_negative_exemplars = 9244 * 3
# 	The first step is to initialize weights for all the examples in the training set
# 	    for every image in train_data:
# 			given_label = data[1]
# 			if given_label == correct_label:
# 				weights[image] = 1 / (2 * float(num_of_positive_exemplars))
# 			else:
# 				weights[image] = 1 / (2 * float(num_of_negative_exemplars))
# 	For t in range(0,T)
# 		weights = normalize(weights)
# 		for stump in decision_stumps:
# 			if (stump not in selected_stumps):
# 				hypothesis[stump] = learning(train_data, weights, stump, correct_label)
# 				error = 0
# 				for image, data in train_data.items():
# 					actual_label = correct_label if data[1] == correct_label else incorrect_label
# 					assigned_label = hypothesis[stump][image]
# 					if actual_label != assigned_label:
# 						error = error + weights[image]
# 				if error < min_error:
# 					min_error = error
# 					best_stump = stump
# 		selected_stumps.append(best_stump)  ==> pick best stump and its alpa value
# 		compute beta as beta = min_error / (1 - min_error)
# 		compute alpha as math.log(1 / beta)
# 		z_alpha_values.append(alpha)
# 	return selected stumps and corresponding alpha values

#
# The alpha values correspond to the confidence value of that stump
#
# We do this for all classifiers (0,90,180,270) and get 4 lists of (stumps, alpha)
#
# Now for each classifier we test our files in testing set.
# Whichever classifier returns maximum weight, we assign that as the predicted orientation
#
#
# Assumptions made:
# Precomputed some values that seemed promising
# Also, instead of using (192*192) values, we are using random 50 values

import collections
import sys
import numpy as np
from random import random, randint
import math

#for KNN:
train_data = []
test_data = []

#for NNET:
hidden_neurons = 31 #91,0.01,35%,1  31,0.01,45.1%,1  21,0.01,42.65,1  19,0.01,34.26,1  19,0.01,23.43,2  31,0.1,Wd10,46,1
                    #31,0.1,Wd15,43,1  #29,0.1,Wd20,
B = 1                # BIAS  [0.0001]*hidden_neurons
hidden_layers = 1
out = 4                     #num of classes
num_epochs = 40
ALPHA = 0.01  #0.001 = 32
decay_rate = 1.0


#for ADAB:




#------------> NNET Implementation------------------>
def read_file_nn(filename):
    print "READING FILE :",filename, "---->"
    f = open(filename, 'r')
    content = f.readline()
    images = {}
    #img_list = []
    while content != '':
        name,value = content.split(".jpg ")
        content = f.readline()  #open(filename).read().splitlines()
        orien_val = map(int, value.split(" "))
        orien = orien_val[0]
        ascii_val = orien_val[1:]
        if sum(ascii_val) != 0:
            normal_ascii_val = [ele/255.0 for ele in ascii_val]
        images[name+".jpg_"+str(orien)] = normal_ascii_val
    return images, len(ascii_val)


def sigmoid_nn(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    #x -= np.max(x)
    return 1/ (1 + np.exp(-x))


'''def create_synapses(r,c):
    np.random.seed(1)
    return 2 * np.random.random((r, c)) - 1'''


def store_synapses_nn(syn, mfile):
    f = open(mfile, "w")
    res = [syn[i] for i in range(0,hidden_layers+1)]
    np.save(f, res)


def load_synapses_nn(file_name):
    syn = []
    lis = np.load(file_name)
    layer = 0
    while layer < len(lis):
        syn.append(lis[layer])
        layer += 1
    return syn


def create_syn_nn(l):
    synapse = []
    synapse.append(np.random.normal(0, 0.1, (hidden_neurons, l)))
    #for i in range(1,hidden_layers):
    #    synapse.append(np.random.normal(0, 0.1, (hidden_neurons, hidden_neurons)))
    #synapse.append(np.random.normal(0, 0.1, (73, hidden_neurons)))
    synapse.append(np.random.normal(0, 0.1, (out, hidden_neurons)))
    return synapse


def forward_prop_nn(nn, synapse):
    x = []
    layers = []
    layers.append(nn["input"])
    for i in range(0,hidden_layers+1):
        layers.append(sigmoid_nn(synapse[i].dot(layers[i])))
    return layers


def softmax_nn(y):
    y -= np.max(y)
    lis = np.exp(y) / np.sum(np.exp(y))
    return lis


def back_prop_nn(layers, nn, syn_backp):
    error = []
    delta = []
    error.append(softmax_nn(layers[-1]) - nn["output"])
    i = -1
    j = 0
    while abs(i) < len(layers):
        delta.append(error[j] * sigmoid_nn(layers[i], True))
        #error.append(delta[j].T.dot(synapse[i]))
        error.append(syn_backp[i].T.dot(delta[j]))
        i -= 1
        j += 1

    ## UPDATING THE WEIGHTS
    j = -1
    for i in range(0, len(syn_backp)):
        syn_backp[i] += ALPHA * (delta[j].dot(layers[i].T))
        j -= 1

    return syn_backp, error


def train_nn(nn, synapse):
    layers = forward_prop_nn(nn, synapse)
    synapse, err = back_prop_nn(layers, nn, synapse)
    return synapse, layers[-1], err


def solve_nnet(action, image_file, model_file):
    global ALPHA
    input_images, l = read_file_nn(image_file)
    nnet = {}
    synapses = create_syn_nn(l)
    map = 0.0
    optimum_weights = synapses
    if action == 'train':
        print "STARTING TRAINING.."
        keys = input_images.keys()
        accuracy = 0.0
        prev_error = 0.1
        curr_error = 0.01
        iter = 0
        #for iter in range (1, num_epochs+1):
        while iter <= num_epochs: #and (prev_error - curr_error)/prev_error >= 0.0001:
            #synapses = optimum_weights
            random.shuffle(keys)
            iter += 1
            all_image_errors = [100.0]
            matched = 0
            if iter%10 == 0: #and iter <= 30:  # Implementing alpha decay
                ALPHA = ALPHA/decay_rate
            if iter == 30:
                ALPHA *= 1.0
            for image in keys:
                #print " \n\n processing -",image
                input_data = input_images[image]
                img_name, ground_truth = image.split("_")
                output = [0, 0, 0, 0]
                output[int(ground_truth)/90] = 1
                nnet['input'] = np.reshape(input_data,(l,1))
                nnet['output'] = np.reshape(output,(out,1))
                synap, pred, error = train_nn(nnet, synapses)
                if np.mean(np.abs(error[-1])) < all_image_errors[-1]:
                    synapses = synap
                    all_image_errors.append(np.mean(np.abs(error[-1])))
                if str((pred.tolist().index(max(pred.tolist())))*90) == ground_truth:
                    matched +=1
            #print [np.shape(e) for e in synapses]
            print "\tACCURACY:", matched, "of",len(input_images), (matched/float(len(input_images)))*100,"%"
            if matched/float(len(input_images)) > map:
                map = matched/float(len(input_images))
                optimum_weights = synapses
            prev_error = curr_error
            curr_error = np.mean(np.abs(all_image_errors))
            print "\tmean error for", str(iter), "iterations:", curr_error
        print "STORING PARAMETERS in file..."
        store_synapses_nn(optimum_weights,model_file)

    else:
        print "STARTING TESTING.."
        print "LOADING PARAMETERS.."
        matched = 0
        f = open("output.txt", "w")
        synapses = load_synapses_nn(model_file)
        for image in input_images:
            print " \n\n testing -",image
            input_data = input_images[image]
            img_name, ground_truth = image.split("_")
            nnet['input'] = np.reshape(input_data,(l,1))
            lay = forward_prop_nn(nnet, synapses)
            p = softmax_nn(lay[-1]).tolist()
            max_predict_orient = p.index(max(p))*90  #(p.index(max(p)))*90
            print "max_predict_orient:", max_predict_orient, ground_truth
            if max_predict_orient == int(ground_truth):
                matched += 1
            print "IMAGE:", image.split('_')[0], "PREDICTION:", max_predict_orient, "GROUND_TRUTH:", ground_truth
            f.write(image.split('_')[0] + " " + str(max_predict_orient) + "\n")
        print "ACCURACY:", (matched / float(len(input_images)))*100.0,"%"



##------------> KNN Implementation------------------>

def read_data(inputfile):
    pixels = []
    file = open(inputfile, 'r');
    for line in file:
        data = tuple([val for val in line.split()])
        pixels += [(data), ]
    return pixels


def get_distance(image1,image2):
    dist = np.sum(np.linalg.norm(image2.reshape((64, 3)) - image1.reshape((64, 3)), axis=1))
    return dist

def get_all_distances(test_dat,train_dat):
    # distances = []

    distances = [(train_dat[i][1],
                  get_distance(test_dat, np.array(train_dat[i][2::]).astype(np.float32)))
                  for i in range(len(train_dat))]
    return distances

def get_sorted_distances(distances):
    sorted_distances = sorted(distances, key=lambda tup: tup[1])
    # sorted_distances = np.lexsort(distances[1],distances)
    # sorted_dist = sorted_distances[0:7]
    return sorted_distances

def get_k_nearest_neighbours(sorted_distances):
    # neighbours = []
    # sorted_dist = sorted_distances[0:k]
    neighbours = [sorted_distances[i][0] for i in range(len(sorted_distances))]
    return neighbours

def print_predicted_orient(predicted_postion,img_name):
    print "Predicted orientation of image '"+img_name+"' is: ",
    if (predicted_postion == '0'):
        print "upright (North)."
    elif (predicted_postion == '90'):
        print "rotated to the right (East)."
    elif (predicted_postion == '180'):
        print "upside down (South)."
    elif (predicted_postion == '90'):
        print "rotated to the left (West)."
    return

def write_output(all_positions, predicted_positions,img):
    f = open('output_nearest.txt','w')
    for i in range(0,len(all_positions)):
        res=str(img[i])+str(all_positions[i])+str(predicted_positions[i])
        f.write(res)


####################
# Main function for KNN

def solve_knn(action, image_file, model_file):
    global train_data
    global test_data
    mf = open(model_file, 'r');
    k = int(mf.readline())  # Reading expected value of "k" from the file
    if action == 'train':
        print "Reading Input Data - "
        train_data = read_data(image_file)

    else:
        test_data = read_data(image_file)
        img_names = [test_data[i][0] for i in range(len(test_data))]
        all_positions = [test_data[i][1] for i in range(len(test_data))]
        predicted_positions = []

        for i in range(len(test_data)):
            test_dat = np.array(test_data[i][2::]).astype(np.float32)
            distances = get_all_distances(test_dat,train_data)
            sorted_distances = get_sorted_distances(distances)
            positions = get_k_nearest_neighbours(sorted_distances[0:k])
            predicted_position = collections.Counter(positions).most_common(1)[0][0]
            predicted_positions += [predicted_position, ]
            # Ref for finding most frequent occurrence :
            # https://stackoverflow.com/questions/6987285/python-find-the-item-with-maximum-occurrences-in-a-list
            # SORTING : https://stackoverflow.com/questions/3121979/how-to-sort-list-tuple-of-lists-tuples
        write_output(all_positions,predicted_positions,img_names)
        print "Ground truth: ",all_positions
        print "Predicted: ",predicted_positions
        accuracy = float(len([ a for a,b in zip(all_positions,predicted_positions) if a==b ]))/float(len(all_positions))*100
        print "Accuracy: ",accuracy,"%"


#### Solve for adaboost

def read_data_adaboost(inputfile):
    images = {}
    file = open(inputfile, 'r');
    for line in file:
        name, value = line.split(".jpg ")
        orien_val = map(int, value.split(" "))
        orien = orien_val[0]
        pixels = orien_val[1:]
        name_orientation = name + "_" + str(orien)
        images[name_orientation] = (pixels, str(orien))
    return images

def read_data_adaboost_test(inputfile):
    images = {}
    file = open(inputfile, 'r');
    for line in file:
        name, value = line.split(".jpg ")
        orien_val = map(int, value.split(" "))
        orien = orien_val[0]
        pixels = orien_val[1:]
        name_orientation = name + ".jpg"
        images[name_orientation] = (pixels, str(orien))
    return images

def learning(examples, weights, (i, j), label):
    h = dict()
    for image, data in examples.items():
        pixels = data[0]
        if pixels[i] > pixels[j]:
            h[image] = label
        else:
            h[image] = "Other"
    return h


def normalize(weights):
    normalized_weights = dict()
    total = 0
    for image, w in weights.items():
        total += w
    for image, w in weights.items():
        normalized_weights[image] = (w / float(total))
    return normalized_weights


def adaboost_0(train_data, decision_stumps):
    decision_stumps = [(11, 191), (17, 146), (2, 122), (5, 43), (23, 76), (145, 188), (182, 11), (20, 140), (121, 167),
                       (142, 170),
                       (191, 25), (29, 115), (139, 173), (50, 143), (188, 2), (5, 46), (47, 5), (26, 143), (191, 26),
                       (145, 191)] + decision_stumps
    num_of_exemplars = len(train_data)
    num_of_positive_exemplars = num_of_exemplars / 4
    num_of_negative_exemplars = num_of_positive_exemplars * 3

    weights = dict()
    correct_label = "0"
    incorrect_label = "Other"

    for image, data in train_data.items():
        given_label = data[1]
        if given_label == correct_label:
            weights[image] = 1 / (2 * float(num_of_positive_exemplars))
        else:
            weights[image] = 1 / (2 * float(num_of_negative_exemplars))

    T = 20
    hypothesis = dict()
    selected_stumps = list()
    z = list()
    for t in range(0, T):
        weights = normalize(weights)
        min_error = 1000;
        best_stump = (-1, -1)
        for stump in decision_stumps:
            if (stump not in selected_stumps):
                hypothesis[stump] = learning(train_data, weights, stump, correct_label)
                error = 0
                for image, data in train_data.items():
                    actual_label = correct_label if data[1] == correct_label else incorrect_label
                    assigned_label = hypothesis[stump][image]
                    if actual_label != assigned_label:
                        error = error + weights[image]
                if error < min_error:
                    min_error = error
                    best_stump = stump
        selected_stumps.append(best_stump)
        # update weights
        updated_weights = dict()
        beta = min_error / (1 - min_error)
        for image, data in train_data.items():
            actual_label = correct_label if data[1] == correct_label else incorrect_label
            assigned_label = hypothesis[best_stump][image]
            if actual_label == assigned_label:
                updated_weights[image] = weights[image] * beta
            else:
                updated_weights[image] = weights[image]
        weights = updated_weights
        alpha = math.log(1 / beta)
        z.append(alpha)
    return selected_stumps, z


def adaboost_90(train_data, decision_stumps):
    decision_stumps = [(95, 170), (143, 5), (47, 32), (188, 176), (81, 2), (191, 149), (160, 2), (167, 152), (136, 2),
                       (47, 2), (23, 163), (191, 34), (47, 151), (185, 74), (23, 50), (167, 145), (112, 5),
                       (163, 2)] + decision_stumps
    num_of_exemplars = len(train_data)
    num_of_positive_exemplars = num_of_exemplars / 4
    num_of_negative_exemplars = num_of_positive_exemplars * 3

    weights = dict()
    correct_label = "90"
    incorrect_label = "Other"

    for image, data in train_data.items():
        given_label = data[1]
        if given_label == correct_label:
            weights[image] = 1 / (2 * float(num_of_positive_exemplars))
        else:
            weights[image] = 1 / (2 * float(num_of_negative_exemplars))

    T = 20
    hypothesis = dict()
    selected_stumps = list()
    z = list()
    for t in range(0, T):
        weights = normalize(weights)
        min_error = 1000;
        best_stump = (-1, -1)
        for stump in decision_stumps:
            if (stump not in selected_stumps):
                hypothesis[stump] = learning(train_data, weights, stump, correct_label)
                error = 0
                for image, data in train_data.items():
                    actual_label = correct_label if data[1] == correct_label else incorrect_label
                    assigned_label = hypothesis[stump][image]
                    if actual_label != assigned_label:
                        error = error + weights[image]
                if error < min_error:
                    min_error = error
                    best_stump = stump
        selected_stumps.append(best_stump)
        # update weights
        updated_weights = dict()
        beta = min_error / (1 - min_error)
        for image, data in train_data.items():
            actual_label = correct_label if data[1] == correct_label else incorrect_label
            assigned_label = hypothesis[best_stump][image]
            if actual_label == assigned_label:
                updated_weights[image] = weights[image] * beta
            else:
                updated_weights[image] = weights[image]
        weights = updated_weights
        alpha = math.log(1 / beta)
        z.append(alpha)
    return selected_stumps, z


def adaboost_180(train_data, decision_stumps):
    decision_stumps = [(182, 2), (176, 47), (185, 177), (179, 181), (27, 41), (36, 53), (110, 86),(188, 20), (169, 76),
                       (185, 53), (179, 68)] + decision_stumps
    num_of_exemplars = len(train_data)
    num_of_positive_exemplars = num_of_exemplars / 4
    num_of_negative_exemplars = num_of_positive_exemplars * 3

    weights = dict()
    correct_label = "180"
    incorrect_label = "Other"

    for image, data in train_data.items():
        given_label = data[1]
        if given_label == correct_label:
            weights[image] = 1 / (2 * float(num_of_positive_exemplars))
        else:
            weights[image] = 1 / (2 * float(num_of_negative_exemplars))

    T = 20
    hypothesis = dict()
    selected_stumps = list()
    z = list()
    for t in range(0, T):
        weights = normalize(weights)
        min_error = 1000;
        best_stump = (-1, -1)
        for stump in decision_stumps:
            if (stump not in selected_stumps):
                hypothesis[stump] = learning(train_data, weights, stump, correct_label)
                error = 0
                for image, data in train_data.items():
                    actual_label = correct_label if data[1] == correct_label else incorrect_label
                    assigned_label = hypothesis[stump][image]
                    if actual_label != assigned_label:
                        error = error + weights[image]
                if error < min_error:
                    min_error = error
                    best_stump = stump
        selected_stumps.append(best_stump)
        # update weights
        updated_weights = dict()
        beta = min_error / (1 - min_error)
        for image, data in train_data.items():
            actual_label = correct_label if data[1] == correct_label else incorrect_label
            assigned_label = hypothesis[best_stump][image]
            if actual_label == assigned_label:
                updated_weights[image] = weights[image] * beta
            else:
                updated_weights[image] = weights[image]
        weights = updated_weights
        alpha = math.log(1 / beta)
        z.append(alpha)
    return selected_stumps, z


def adaboost_270(train_data, decision_stumps):
    decision_stumps = [(98, 23),(50,188), (146,164), (170, 179), (18,191), (146,109), (184, 185), (77, 39)] + decision_stumps
    num_of_exemplars = len(train_data)
    num_of_positive_exemplars = num_of_exemplars / 4
    num_of_negative_exemplars = num_of_positive_exemplars * 3

    weights = dict()
    correct_label = "270"
    incorrect_label = "Other"

    for image, data in train_data.items():
        given_label = data[1]
        if given_label == correct_label:
            weights[image] = 1 / (2 * float(num_of_positive_exemplars))
        else:
            weights[image] = 1 / (2 * float(num_of_negative_exemplars))

    T = 20
    hypothesis = dict()
    selected_stumps = list()
    z = list()
    for t in range(0, T):
        weights = normalize(weights)
        min_error = 1000;
        best_stump = (-1, -1)
        for stump in decision_stumps:
            if (stump not in selected_stumps):
                hypothesis[stump] = learning(train_data, weights, stump, correct_label)
                error = 0
                for image, data in train_data.items():
                    actual_label = correct_label if data[1] == correct_label else incorrect_label
                    assigned_label = hypothesis[stump][image]
                    if actual_label != assigned_label:
                        error = error + weights[image]
                if error < min_error:
                    min_error = error
                    best_stump = stump
        selected_stumps.append(best_stump)
        # update weights
        updated_weights = dict()
        beta = min_error / (1 - min_error)
        for image, data in train_data.items():
            actual_label = correct_label if data[1] == correct_label else incorrect_label
            assigned_label = hypothesis[best_stump][image]
            if actual_label == assigned_label:
                updated_weights[image] = weights[image] * beta
            else:
                updated_weights[image] = weights[image]
        weights = updated_weights
        alpha = math.log(1 / beta)
        z.append(alpha)
    return selected_stumps, z


def store_para(lis1, lis2, lis3, lis4, lis5, lis6, lis7, lis8, model_file):
    f = open(model_file, 'w')
    res = [lis1, lis2, lis3, lis4, lis5, lis6, lis7, lis8]
    np.save(f, res)

def load_params(fname):
    f = open(fname, 'r')
    arr = np.load(fname)
    lis = arr.tolist()
    return (lis[0], lis[1], lis[2], lis[3], lis[4], lis[5], lis[6], lis[7])


def compute_h_t((i, j), pixels, label):
    if pixels[i] > pixels[j]:
        return label
    else:
        return "Other"

def test_classifier_0(decision_stumps, z, data):
    total_stumps = len(decision_stumps)
    pixels = data[0]
    assigned_label = "0" if data[1] == "0" else "Other"
    sum_1 = 0
    sum_2 = 0
    for i in range(0, total_stumps):
        stump = decision_stumps[i]
        alpha = z[i]
        h_t = compute_h_t(stump, pixels, "0")
        if h_t == assigned_label:
            sum_1 += alpha
        sum_2 += alpha
    return sum_1


def test_classifier_90(decision_stumps, z, data):
    total_stumps = len(decision_stumps)
    pixels = data[0]
    assigned_label = "90" if data[1] == "90" else "Other"
    sum_1 = 0
    sum_2 = 0
    for i in range(0, total_stumps):
        stump = decision_stumps[i]
        alpha = z[i]
        h_t = compute_h_t(stump, pixels, "90")
        if h_t == assigned_label:
            sum_1 += alpha
        sum_2 += alpha
    return sum_1


def test_classifier_180(decision_stumps, z, data):
    total_stumps = len(decision_stumps)
    pixels = data[0]
    assigned_label = "180" if data[1] == "180" else "Other"
    sum_1 = 0
    sum_2 = 0
    for i in range(0, total_stumps):
        stump = decision_stumps[i]
        alpha = z[i]
        h_t = compute_h_t(stump, pixels, "180")
        if h_t == assigned_label:
            sum_1 += alpha
        sum_2 += alpha
    return sum_1


def test_classifier_270(decision_stumps, z, data):
    total_stumps = len(decision_stumps)
    pixels = data[0]
    assigned_label = "270" if data[1] == "270" else "Other"
    sum_1 = 0
    sum_2 = 0
    for i in range(0, total_stumps):
        stump = decision_stumps[i]
        alpha = z[i]
        h_t = compute_h_t(stump, pixels, "270")
        if h_t == assigned_label:
            sum_1 += alpha
        sum_2 += alpha
    return sum_1


def solve_ada(action, image_file, model_file):

    if action == "train":
        train_data = read_data_adaboost(image_file)

        decision_stumps = list()
        for i in range(0, 50):
            pixel1 = -1
            pixel2 = -1
            while pixel1 == -1 or pixel2 == -1 or (pixel1, pixel2) in decision_stumps:
                pixel1 = randint(0, 191)
                pixel2 = randint(0, 191)
            decision_stumps.append((pixel1, pixel2))

        (stumps_0, z_0) = adaboost_0(train_data, decision_stumps)

        decision_stumps = list()
        for i in range(0, 50):
            pixel1 = -1
            pixel2 = -1
            while pixel1 == -1 or pixel2 == -1 or (pixel1, pixel2) in decision_stumps:
                pixel1 = randint(0, 191)
                pixel2 = randint(0, 191)
            decision_stumps.append((pixel1, pixel2))

        (stumps_90, z_90) = adaboost_90(train_data, decision_stumps)

        decision_stumps = list()
        for i in range(0, 50):
            pixel1 = -1
            pixel2 = -1
            while pixel1 == -1 or pixel2 == -1 or (pixel1, pixel2) in decision_stumps:
                pixel1 = randint(0, 191)
                pixel2 = randint(0, 191)
            decision_stumps.append((pixel1, pixel2))
        (stumps_180, z_180) = adaboost_180(train_data, decision_stumps)

        decision_stumps = list()
        for i in range(0, 50):
            pixel1 = -1
            pixel2 = -1
            while pixel1 == -1 or pixel2 == -1 or (pixel1, pixel2) in decision_stumps:
                pixel1 = randint(0, 191)
                pixel2 = randint(0, 191)
            decision_stumps.append((pixel1, pixel2))
        (stumps_270, z_270) = adaboost_270(train_data, decision_stumps)


        store_para(stumps_0, z_0, stumps_90, z_90, stumps_180, z_180, stumps_270, z_270,model_file)
    else:
        test_data = read_data_adaboost_test(image_file)
        (stumps_0, z_0, stumps_90, z_90, stumps_180, z_180, stumps_270, z_270) = load_params(model_file)
        print "STARTING TESTING.."
        print "LOADING PARAMETERS.."
        f = open("output.txt", "w")
        for image, data in test_data.items():
            # actual = data[1]
            weight_0 = test_classifier_0(stumps_0, z_0, data)

            weight_90 = test_classifier_90(stumps_90, z_90, data)

            weight_180 = test_classifier_180(stumps_180, z_180, data)

            weight_270 = test_classifier_270(stumps_270, z_270, data)
            arr = [weight_0, weight_90, weight_180, weight_270]
            max_value = max(arr)
            predicted = arr.index(max_value)
            if predicted == 0:
                predicted = "0"
            elif predicted == 1:
                predicted = "90"
            elif predicted == 2:
                predicted = "180"
            elif predicted == 3:
                predicted = "270"
            f.write(image + " " + str(predicted) + "\n")

#Main function of orient01.py

def solve(action, image_file, model_file, model):
    if str(model) == 'nnet' or 'nnet' in model:
        solve_nnet(action, image_file, model_file)

    elif str(model) == 'nearest' or 'nearest' in model or str(model) == 'best' or 'best' in model:
        solve_knn(action, image_file, model_file)

    elif str(model) == 'adaboost' or 'adaboost' in model:
        solve_ada(action, image_file, model_file)

    else:
        solve_knn(action, image_file, model_file)


if len(sys.argv) < 4:
    print "Please enter the training and test file name as follows : "
    print "    ./program.py train imagefile modelfile model"
    sys.exit()

action = sys.argv[1]
image_file = sys.argv[2]
model_file = sys.argv[3]
model = sys.argv[4]
solve(action, image_file, model_file,model)


