#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Vinita Boolchandani
# (based on skeleton code by D. Crandall, Oct 2017)
#
'''With this code I will be predicting a sequence from an image using three methods:
Using the test image data, train image data and train text data I am trying to calculate three probabilities here:
A. Initial Probability:
    Initial Probability for letter[i] is calculated as the number of statement starting from letter[i] over total number of statements in file.
B. Transition Probability:
    Transition probability is calculated from the text training file by counting the transitions of letter[i] to letter[j] divided by the total transitions of letter[i]
    this gives me P(l[j]|l[i]) where j represents the next alphabet in unobserved sequence and i is the current alphabet
C. Emission Probability E:
    This E represents the probability of l(i) = letters[j] given O(i) = test_letters[r]
    where l(i) is the i-th element of sequence to be predicted, O(i) is the observed variable which is currently set to the r-th pixel arrangement of test_letters

Using these three probabilities the prediction is done in three ways:
I. Simplified method:
In this method,
1. I will be calculating the probability by taking each observed alphabet O from test sequence
2. Then Each pixel of this observed alphabet O is matched with each of the train alphabets in 'letters'
3. Based on number of pixels matched, I am calculating the emission probability E for each alphabet in letters.
4. The maximum value of this emission probability E gives the most probable alphabet
5. This sequence repeated for each alphabet gives the most probable sequence.

II. HMM using variable elimination:
In this method,
1. let the sequence to be predicted be l1,l2,l3,...,ln
2. The observed pixels states be O1,O2,O3...,On
3. Here the probability for l1 is found for all letters in LETTERS
4. This is done by multiplying the initialProb(l1) by emission_prob(l1|O1)
5. Then l1 is eliminated by marginalising all l1 terms over all values of l1
6. l1 terms include P(l1)*P(l2|O2)*P(l2|l1). This can be replaced by alpha(l2)
7. Similarly eliminating l2 terms and l3,l4,...ln terms will give us a lookup table with l1,l2,...ln values
8. Selecting the maximum probability values gives the maximum probable sequence

III. HMM using Viterbi Algorithm:
In this method,
1. let the sequence to be predicted be l1,l2,l3,...,ln
2. The observed pixels states be O1,O2,O3...,On
3. Here the probability for l1 is found for all letters in LETTERS
4. This is done by multiplying the initialProb(l1) by emission_prob(l1|O1)
5. The max probability from l1 values is taken as the initial probability for predicting l2 value
6. probability of l2 is calculated as,
    P(l2) = max(initial_prob(l1)*P(l2|l1)*P(l2|O2))
7. This max value of P(l2) is used as initial probability to calculate P(l3) and so on.
8. At each step only the maximum probability value moves forward to the next step.
9 At each step the maximum prob alphabet is the predicted alphabet of that test sequence step.
10. Hence at the end we get the most probable sequence.
11. This process takes less time than variable elimination

'''
from PIL import Image #, ImageDraw, ImageFont
import sys
import math

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25
#GROUND_TRUTH = "GINSBURG, BREYER, SOTOMAYOR, and KAGAN, JJ., joined."


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result


def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }


def read_data(fname):
    words_corpus = []
    file = open(fname, 'r')
    for line in file:
        words_corpus.append(line)
    return words_corpus


def calc_initial_prob(train_data):
    initial_prob = {}
    n = len(train_data)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    for alpha in letters:
        initial_prob[alpha] = len(filter(lambda x: x.startswith(alpha),train_data))
        if initial_prob[alpha] > 0:
            initial_prob[alpha] = math.log(initial_prob[alpha]) - math.log(float(n))
    return initial_prob


def calc_transition_prob(train_data):
    transition_prob = {}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    for alpha in letters:
        transition_prob[alpha] = [0.0]*len(letters)
    for ele in train_data:
        w = list(ele)
        for i in range(1,len(w)):
            first = w[i-1]
            second = w[i]
            if first in transition_prob.keys() and second in letters:
                transition_prob[first][letters.index(second)] += 1
    for alpha in letters:
        total_trans = float(sum(transition_prob[alpha]))
        for i in range(0, len(letters)):
            if transition_prob[alpha][i] > 0:
                transition_prob[alpha][i] = math.log(transition_prob[alpha][i]) - math.log(total_trans) #transition_prob[alpha][i]/total_trans #math.fabs(math.log(transition_prob[alpha][i]) - math.log(total_trans))
    return transition_prob


def train_priors(train_data):
    # This function calculates P(p1,p2,p4,p5|letter = 'a') which the probability of pixels being ON given the letter is 'a'.
    # This will be calculated for all LETTERS being considered and their given pixel arrangements
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    total = 0
    priors = [0.1]*len(letters)
    for ele in train_data:
        for i in list(ele):
            if i in letters:
                pos = letters.index(i)
                priors[pos] += 1
                total += 1
    res = [math.log(j) - math.log(float(total)) for j in priors]
    return res


def calc_emission_prob_nb(train_letters, state_pixel, priors):
    res = []
    total = math.fabs(math.log(0.9))*CHARACTER_HEIGHT*CHARACTER_WIDTH
    noise = 0.99
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    for a in letters:
        if a in train_letters:
            prod = 0.0
            prod2 = 0.0
            matched_blacks = 0
            matched = 0
            train_image = train_letters[a]
            for i in range(0,len(state_pixel)):
                for j in range(0,len(state_pixel[i])):
                    if state_pixel[i][j] == train_image[i][j]:
                        matched += 1
                        prod += math.log(noise)
                    else:
                        prod += math.log(1 - noise)
                    '''if state_pixel[i][j] == '*' and train_image[i][j] == '*':
                        matched_blacks += 1
                        prod += math.fabs(math.log(noise))
                    elif state_pixel[i][j] == ' ' and train_image[i][j] == ' ':
                        matched_blacks += 1
                        prod2 += math.fabs(math.log(noise))'''
            #print priors[letters.index(a)]
            #if priors[letters.index(a)] >= 0.0:
            prod += priors[letters.index(a)]
            res.append(prod)
    #print res
    #if max(res) <= 1.0:
     #   res[-1] = 1.0
    return res


def hmm_ve(train_letters, test_let, train_data):
    # using forward inference:
    # According to fig 1(b) we don't have any dependencies among the states. So transition probabilities won't exist.
    # Hence P(l(t+1),O(1),O(2),..O(t+1)) = alpha(t+1) * (l(t+1)) = sum{alpha(t) * l(t) * P(O(t+1)|l(t+1))}
    # Where alpha(t) is introduced after variable elimination of t, t-1 and so on recursively

    initial_p = calc_initial_prob(train_data)
    prior = train_priors(train_data)
    predicted_seq = ''
    transition_p = calc_transition_prob(train_data)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    p = [[0] * len(letters)] * len(test_let)
    emission_prob = calc_emission_prob_nb(train_letters, test_let[0], prior)
    p[0] = [emission_prob[s] + initial_p[letters[s]] for s in range(0, len(letters))]
    max_p = max(p[0])
    pos = p[0].index(max_p)
    predicted_seq += letters[pos]
    for r in range(1, len(test_let)):
        e = calc_emission_prob_nb(train_letters, test_let[r], prior)
        for i in range(0, len(letters)):
            temp = [0] * len(letters)
            for j in range(0,len(letters)):
                #print p[r-1][j], e[i], transition_p[letters[j]][i], "==", p[r-1][j] + e[i] + transition_p[letters[j]][i]
                temp[j] =  e[i] + transition_p[letters[j]][i] #+ p[r-1][j]/float(len(letters)) #transition_p[letters[j]][i] +p[r-1][j] #++ e[i]
            p[r][i] = sum(temp)#/float(len(letters))
        max_p = max(p[r])
        if max_p == 0:
            print p[r]
        #print max_p, p[r].index(max_p)
        pos = p[r].index(max_p)
        #print p[r], max(p[r]), letters[pos]
        predicted_seq+=letters[pos]
    #matched = 0
    #i = 0
    #str = GROUND_TRUTH
    #for ele1 in str:  # range(0,len(str)):
    #    if ele1 == predicted_seq[i]:
    #        matched += 1
    #    i += 1
    #print "VE Accuracy:", (matched / float(len(str))) * 100, "%"
    return predicted_seq


def simplified(train_letters, test_let):
    initial_p = calc_initial_prob(train_data)
    prior = train_priors(train_data)
    predicted_seq = ''
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    p = [[0] * len(letters)] * len(test_let)
    emission_prob = calc_emission_prob_nb(train_letters, test_let[0], prior)
    p[0] = [emission_prob[s] + initial_p[letters[s]] for s in range(0, len(letters))]
    p1 = max(p[0])
    pos = p[0].index(p1)
    predicted_seq += letters[pos]
    for r in range(1, len(test_let)):
        e = calc_emission_prob_nb(train_letters, test_let[r], prior)
        for j in range(0, len(letters)):
            p2 = e[j]
            p[r][j] = p1 + p2
        p1 = max(p[r])
        pos = p[r].index(p1)
        predicted_seq += letters[pos]
    return predicted_seq


def hmm_viterbi(train_letters, test_let, train_data):
    prior = train_priors(train_data)
    x = ''
    transition_p = calc_transition_prob(train_data)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    initial_p = calc_initial_prob(train_data) #[1 / float(len(letters))] * len(letters)
    p = [[0]*len(letters)]*len(test_let)
    emiss = calc_emission_prob_nb(train_letters, test_let[0], prior)
    p[0] = [emiss[s] + initial_p[letters[s]] for s in range(0, len(letters))]
    p1 = max(p[0])
    pos = p[0].index(p1)
    x += letters[pos]
    #print x
    for r in range(1,len(test_let)):
        e = calc_emission_prob_nb(train_letters, test_let[r], prior)
        for j in range(0,len(letters)):
            p2 = e[j]
            p3 = transition_p[letters[pos]][j]
            #print p1, p2, p3, "==" ,p1+p2+p3
            p[r][j] = p1 + p2 + p3 #math.fabs(math.log(p1))
        p1 = max(p[r])
        pos = p[r].index(p1)
        #print letters[pos]
        x += letters[pos]
    #matched = 0
    #i = 0
    #str = GROUND_TRUTH
    #for ele1 in str:#range(0,len(str)):
    #    if ele1 == x[i]:
    #        matched += 1
    #    i += 1
    #matched = len(filter(lambda y: y in list(str), list(x)))
    #print "Viterbi Accuracy:", (matched / float(len(str))) * 100, "%"
    return x


def solve(train_data, train_letters, test_let):
    #x = hmm_VE(train_letters, test_let)
    z = hmm_viterbi(train_letters, test_let, train_data)
    y = simplified(train_letters, test_let)
    x = hmm_ve(train_letters, test_let, train_data)

    print "Simple:", y
    print "HMM VE:", x
    print "HMM Viterbi:", z


#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
#train_img_fname = 'courier-train.png'
#train_txt_fname = 'bc.train'
#test_img_fname = 'test-8-0.png'
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
train_data = read_data(train_txt_fname)
solve(train_data, train_letters, test_letters)

## Below is just some sample code to show you how the functions above work. 
# You can delete them and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
#print "\n".join([ r for r in train_letters['.'] ])

# Same with test letters. Here's what the third letter of the test data
#  looks like:
#print "\n".join([ r for r in test_letters[0] ])
#naive_bayes(train_letters,test_letters)


#print calc_emission_prob_nb(train_likelihood([train_letters['I']]), test_letters[0],1)
#print naive_bayes(train_letters,test_letters)

