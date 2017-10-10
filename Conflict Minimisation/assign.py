#!/usr/bin/env python

'''
1. ABSTRACTION:
--> Initial State: It is taken from the answers entered by the students in the survey, giving the first student as the first preference
--> State Space: All possible team combinations where any team can be of size 3 or less.
    To reduce the state space we have incorporated features like:
    * Visited successors are not revisited
    * One randomly selected student is shuffled in different teams at a time to generate a set of different successors
    * considered only one successor out of the set; the one with least heuristic
--> Successor(s) : Gives all possible states by shuffling one selected student in s to every possible position in other teams.
--> Goal_state: It is undefined for this kind of problems as we want to minimize the heuristic value as much as possible.
    The state with least possible heuristic will be our goal state.
--> Cost Function: In this problem the cost is taken as the heuristic as if we minimize heuristc, we also end up minimizing the cost.
--> Heuristic Function: The heuristic function is k(s) + m(s) + n(s) where s is any given state and k, m, n are costs as given in the Question.
    Therefore, the total heuristic value is the total time spent on each student due to conflicts.
    * This heuristic is admissible as we are always choosing the least heuristic value successor from the set of succesor which means it
      will never overestimate.
2. My algorithm is based on local search:
    --> It generates an initial state from the file. So s = initial state
    --> generate successors s^ for s, by selecting student s at random:
        -> if h(s^) < h(s) and s^ has not been visited already:
              s <- s^
            else,
              -Do the Monte Carlo Descent with prob exp(-(h(s^) - h(s)) / T), s <- s^, where T = 10
    --> return s
3. Assumptions:
--> The input entered by the user is always in the correct format
--> Team size 0 represents no preference
--> Team Member _ represents no preference
--> Heuristic Value is same as the cost or time spent by the AI based on the number of conflicts.
--> Execution needs to be done as: python assign.py [Input file] [K] [M] [N]
--> Assumed the T for Monte Carlo Descent to be 10



'''


#import os
import sys
import random
import numpy


def succ(team):
    ran = random.randint(0,len(students)-1)
    successors = shuffle(students[ran], team, heuristic(team))
    heuristic_vals = list(map(lambda x: heuristic(x), successors))
    successors_sorted = [x1 for _, x1 in sorted(zip(heuristic_vals, successors))]
    return successors_sorted

def shuffle(stu, team, heu_val):
    successor_set = []
    allotted_team = filter(lambda x: stu in x , team)[0]
    index = team.index(allotted_team)
    allotted_team.remove(stu)
    team1 = team[0:index] + team[index+1:]

    for i in range(0, len(team1)):

        #This for loop does the swapping for successor generation
        for j in range(0, len(team1[i])):
            t = team1[0:i]+[team1[i][0:j]+[stu,]+team1[i][j+1:]]+[allotted_team[0:]+[team1[i][j],]] + team1[i+1:]
            if heuristic(t) != heu_val:
                successor_set.append(t)

        # This for loop does the moving(add/remove) for successor generation
        if len(team1[i]) < 3:
            if allotted_team != []:
                t = team1[0:i] + [team1[i][0:0] + [stu, ] + team1[i][0:]] + [allotted_team[0:]] + team1[i + 1:]
            else:
                t = team1[0:i] + [team1[i][0:0] + [stu, ] + team1[i][0:]] + team1[i + 1:]
            if heuristic(t) != heu_val:
                successor_set.append(t)

    #Single case of pulling out the student in a new team without touching other teams
    if allotted_team != []:
        v = team1[0:] + [allotted_team[0:]] + [[stu]]
        successor_set.append(v)
    allotted_team.append(stu)
    return successor_set



def heuristic(team):
    willing_to_work_conflicts = 0
    not_willing_to_work_conflicts = 0
    team_size_conflicts = 0
    for student in students:
        conflict_list = conflicts(team, student)
        willing_to_work_conflicts += conflict_list[0]
        not_willing_to_work_conflicts += conflict_list[1]
        team_size_conflicts += conflict_list[2]
    heu = willing_to_work_conflicts * n + not_willing_to_work_conflicts * m + team_size_conflicts + len(team) * k
    return heu


def conflicts(team, student):
    willing_to_work = 0
    not_willing_to_work = 0
    team_size = 0
    alloted_team = filter(lambda x: student in x, team)[0]
    required_team = file_content[student][1]

    # Conflicts for not getting the preferred team
    if required_team != '_':
        rt = list(required_team.split(','))
        rt = rt + [student,]
        willing_to_work = len(filter(lambda y: y not in alloted_team, rt))

    # Conflicts for getting the unwanted team
    anti_member = file_content[student][2]
    if anti_member != '_':
        anti_member_team = list(anti_member.split(','))
        not_willing_to_work = len(filter(lambda x: x in alloted_team, anti_member_team))

    # Conflicts for unwanted team size
    required_team_size = file_content[student][0]
    if required_team_size != '0':
        if not len(alloted_team) == int(required_team_size):
            team_size = 1

    return [willing_to_work, not_willing_to_work, team_size]


def isgoal(s, c):
    heuris_curr_team = heuristic(c)
    heuris_succ_team = heuristic(s)
    if heuris_curr_team < heuris_succ_team:
        return True
    return False


def solve():
    best_team = curr_team = initial_team
    visited=[curr_team]
    t = 10
    out_count = 1
    while out_count <= 10:
        best_hue = heuristic(best_team)
        curr_team = initial_team
        counter = 0
        flag = False
        while counter<=100: #not isgoal(curr_team, succ_team):
            counter+=1
            succ_list = succ(curr_team)
            for succ_team in succ_list: #succ_team in succ(curr_team):
                if succ_team not in visited:
                    visited.append(succ_team)
                    if len(visited) == 300:
                        visited.pop(0)
                    if heuristic(succ_team) < heuristic(curr_team): #not isgoal(succ_team, curr_team) :
                        curr_team = succ_team[0:]
                    else:
                        flag = True
                        break
                else:
                    succ_list.remove(succ_team)
            if flag:
                diff = (heuristic(succ_list[0]) - heuristic(curr_team)) / (float(t) / counter)
                e = numpy.exp(-1 * diff)
                if e > 0.5:
                    curr_team = succ_list[0]

        if best_hue > heuristic(curr_team):
            best_team = curr_team
        out_count += 1

    return best_team


def readfile(filename):
    f = open(filename, 'r')
    temp1 = open(filename).read().splitlines()
    for str in temp1:
        temp = list(str.lower().split())
        if str != '':
            file_content[temp[0]] = temp[1:]
    start_state = []
    for row in file_content:
        if row not in students:
            t = filter(lambda x: x not in students, file_content[row][1].split(','))
            if len(t) >= 3:
                t = t[0:2]
            t.append(row)
            if '_' in t:
                t.remove('_')
            start_state.append(t)
            for i in t:
                if i not in students:
                    students.append(i)
    return start_state

def printable(t):
    for a in t:
        for b in a:
            print b,
        print ""

initial_team = []
#print "sys.argv[1][0] is", sys.argv
filename = sys.argv[1]

file_content = {}
students = []
initial_team = readfile(filename)
visited = []
k = int(sys.argv[2])
m = int(sys.argv[3])
n = int(sys.argv[4])
result = solve()
printable(result)
print heuristic(result)
