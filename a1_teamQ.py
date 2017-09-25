import sys

import os


def calcCost(conflicts, team):
    cost = 0
    cost = k * len(team) + m*conflicts[0] + n*conflicts[1] + conflicts[2]
    return cost

def succ(team):

    res=[]

    max_heu = 0
    s = ''
    for student in students:
        conflicts = calConflicts(team, student)
        heu = conflicts[0] * m + conflicts[1] * n + conflicts[2]
        if heu > max_heu:
            max_heu = heu   #finding the student whose conflict contributes maximum to the total heuristic value
            s = student
    res = shuffle(s, team)

    #find minimum cost team from res and return that

def shuffle(stu, team):
    new_team = []
    requested_team = file_content[stu][1]
    alloted_team = filter(lambda x: stu in x , team)[0]
    for t in team:
        if len(t) >= 3:
            team.remove(t)
            team.remove(alloted_team)
            alloted_team.remove(stu)
            t1 = [[stu, t[1], t[2] ], alloted_team.append(t[0])]

            t2 = [[t[0], stu, t[2] ], alloted_team.append(t[0])]
            t3 = [t[0], t[1], stu]
    return new_team


def heuristic(team):
    willing_to_work_conflicts = 0
    not_willing_to_work_conflicts = 0
    team_size_conflicts = 0
    for student in students:
        conflicts = calConflicts(team, student)
        willing_to_work_conflicts += conflicts[0]
        not_willing_to_work_conflicts += conflicts[1]
        team_size_conflicts += conflicts[2]
    heu = willing_to_work_conflicts * m + not_willing_to_work_conflicts * n + team_size_conflicts
    return heu

def calConflicts(team, student):
    willing_to_work_conflicts = 0
    not_willing_to_work_conflicts = 0
    team_size_conflicts = 0
    #for student in students:
    alloted_team = filter(lambda x: student in x , team)[0]
    required_team = [file_content[student][1] , student]
    if not alloted_team == required_team:
        willing_to_work_conflicts += 1
    anti_member = file_content[student][2]
    if anti_member in alloted_team:
        not_willing_to_work_conflicts += 1
    required_team_size = file_content[student][0]
    if not len(alloted_team) == required_team_size:
        team_size_conflicts += 1

    return [willing_to_work_conflicts, not_willing_to_work_conflicts, team_size_conflicts]


def isGoal(succ_team, curr_team):
    #curr_team_conflicts = calConflicts(curr_team)  # can be a heuristic function
    #succ_team_conflicts = calConflicts(succ_team)
    heuris_curr_team = heuristic(curr_team)
    heuris_succ_team = heuristic(succ_team)
    if heuris_curr_team < heuris_succ_team:
        return True
    return False

def solve():
    curr_team = initial_team
    succ_team = succ(curr_team)
    if not isGoal(curr_team, succ_team):
        curr_team = succ_team

    #cost = calcCost(conflicts, curr_team)            #has to be minimised
    return curr_team

def readfile(filename):

    f = open(filename, 'r')

    #i = 0
    for line in f:
        temp = list((line.rstrip(os.linesep)).split())
        file_content[temp[0]] = temp[1:]
        #file_content.append(list((line.rstrip(os.linesep)).split()))
        #i += 1
    print file_content
    start_state = []
    for row in file_content:
        if row not in students:
            t=[]
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



initial_team = []
filename = 'Untitled.txt' #int(sys.argv[1])
file_content = {}
students = []
initial_team = readfile(filename)
k = 10 #int(sys.argv[2])
m = 20 #int(sys.argv[3])
n = 30 #int(sys.argv[4])

result = solve()
print("result is:",result)
