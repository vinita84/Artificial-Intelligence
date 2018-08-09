#Assumptions:
## Words which are not present in the dictionary their frequency is taken as 0.001
#Data Cleaning
#All words are converted into the lower case
#Punctuations and empty words have been removed from the data.
#References:
#https://gist.github.com/sebleier/554280
#https://stackoverflow.com/questions/4088265/sorted-word-frequency-count-using-python
#https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python

from __future__ import division
from collections import Counter
from decimal import Decimal
import string
import sys



dict = {}
tweetdict = {}
cities = ['Los_Angeles,_CA','San_Francisco,_CA','Manhattan,_NY','San_Diego,_CA','Houston,_TX',
          'Chicago,_IL','Toronto,_Ontario','Philadelphia,_PA','Atlanta,_GA','Orlando,_FL','Boston,_MA','Washington,_DC']
numberOfTweets = {}
numberOfWordsInTweets = {}
def makeStopWordsDict():
    file1 = open("stop.txt","r")
    stop_words_dict = []
    for line in file1:
        line = line.split()
        stop_words_dict.append(line[0])
    return stop_words_dict

def makeTweetDictionary():
    file1 = open(training_file,"r")
    prevcity = ""
    for line in file1:
        # print(line)
        if line == '\n':
            continue
        line = line.split()
        if line[0] in tweetdict and line[0] in cities:
            tweetdict[line[0]].append(line[1:])
            prevcity = line[0]
        elif line[0] not in tweetdict and line[0] in cities:
            tweetdict[line[0]] = [line[1:]]
            prevcity = line[0]
        else:
            tweetdict[prevcity].append(line)

def makeBagDictionary ():
    file1 = open(training_file,"r")
    stop_words_dict = makeStopWordsDict()
    #print(stop_words_dict)
    prevcity = ""
    for line in file1:
        #print("line is",line)
        if line == '\n':
            #print("empty line found")
            continue
        line = line.split()
        if line[0] in dict and line[0] in cities:
            numberOfTweets[line[0]] = numberOfTweets[line[0]] + 1
            for words in line[1:]:
                words = words.lower()
                words = removePunctuations(words)
                if words not in stop_words_dict and words!= "":
                    dict[line[0]].append(words)
                    numberOfWordsInTweets[line[0]] = numberOfWordsInTweets[line[0]] + 1
            prevcity = line[0]
        elif line[0] not in dict and line[0] in cities:
            numberOfTweets[line[0]] = 1
            dict[line[0]] = []
            numberOfWordsInTweets[line[0]] = 0
            for words in line[1:]:
                words = words.lower()
                words = removePunctuations(words)
                if words not in stop_words_dict and words!="":
                    dict[line[0]].append(words)
                    numberOfWordsInTweets[line[0]] = numberOfWordsInTweets[line[0]] + 1
            prevcity = line[0]
        else:
            #print("prev city is ",prevcity)
            #line = toLower(line)
            for words in line:
                words = words.lower()
                words = removePunctuations(words)
                if words not in stop_words_dict and words!="":
                    dict[prevcity].append(words)
                    numberOfWordsInTweets[prevcity] = numberOfWordsInTweets[prevcity] + 1




def makebags():
    for key in dict:
        txt = dict[key]
        dict[key] = Counter(txt)
        print key,"  ",dict[key].most_common(5)
        #print dict[key]

def toLower(line):
    #print "before con"
    #print line
    line1 = []
    for word in line:
        word = word.lower()
        line1.append(word)
    #print "After conversion"
    #print line1
    return line1

def calculateProb():
    file1 = open(testing_file,"r")
    of = open(output_file,"w")
    test_tweets = 0
    correct_output = 0
    condition_prob_dict = {}
    total_tweets = countTweets(tweetdict)
    for line in file1:
        #print line
        tweet = line
        test_tweets+=1
        line = line.split()
        max = Decimal(0.0)
        final_city = ""
        for city in dict:
            denom = Decimal(numberOfWordsInTweets[city]/numberOfTweets[city])
            #print "deno is",denom
            freq = 1
            conditional_prob = 1
            for word in line[1:]:
                word = removePunctuations(word.lower())
                if word in dict[city]:
                    freq = dict[city][word]
                    #print "freq is ",freq
                    conditional_prob = (conditional_prob* (Decimal(freq/denom)))
                else:
                    freq = Decimal(0.001)
                    conditional_prob = (conditional_prob* (Decimal(freq/denom)))
            #print "condi prob is ",conditional_prob
            final_conditional_prob = (conditional_prob * Decimal(numberOfTweets[city]/total_tweets))
            #print "final cnd prob", final_conditional_prob
            if final_conditional_prob > max:
                max = final_conditional_prob
                #print "inside max",(max)
                final_city = city
        if final_city == line[0]:
            correct_output+=1
        output_string = ""
        #print(line)
        output_string+=(final_city)
        output_string+=(" ")
        output_string+=(line[0])+" "+tweet
        of.write(output_string)
    #print(test_tweets)
    accuracy = float(((correct_output)*100)/test_tweets)
    print "accuracy is ",accuracy


def removePunctuations(word):
    exclude = set(string.punctuation)
    word = ''.join(ch for ch in word if ch not in exclude)

    return word

def countTweets(dict):
    total_number_tweets  = 0
    for key in dict:
        for x in dict[key]:
            total_number_tweets = total_number_tweets + len(dict[key])
    return total_number_tweets


training_file = sys.argv[1]
testing_file = sys.argv[2]
output_file = sys.argv[3]

makeTweetDictionary()
makeBagDictionary()
makebags()
makeStopWordsDict()
countTweets(tweetdict)
# print numberOfTweets
# print numberOfWordsInTweets
calculateProb()

