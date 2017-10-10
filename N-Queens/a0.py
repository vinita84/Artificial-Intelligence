#!/usr/bin/env python
# nrooks.py : Solve the N-Rooks problem!
# D. Crandall, 2016
# Updated by Zehua Zhang, 2017
#
# The N-rooks problem is: Given an empty NxN chessboard, place N rooks on the board so that no rooks
# can take any other, i.e. such that no two rooks share the same row or column.

import sys
import numpy


# Count # of pieces in given row
def count_on_row(board, row):
    return sum(board[row])


# Count # of pieces in given column
def count_on_col(board, col):
    return sum([row[col] for row in board])


# Count total # of pieces on board
def count_pieces(board):
    return sum([sum(row) for row in board])


def count_on_diag(board, row, col):
    b = numpy.asarray(board)
    sum1 = sum(numpy.diagonal(b, col - row, 0, 1))
    if sum1 == 0:
        c = numpy.fliplr(b)
        new_col = N - 1 - col
        sum2 = sum(numpy.diagonal(c, new_col - row, 0, 1))
        if sum2 == 0:
            return True
    return False


# Return a string with the board rendered in a human-friendly format
def printable_board(board):
    if type_name == 'nqueen':
        t = "Q"
    else:
        t = "R"
    return "\n".join([" ".join([t if col == 1 else "_" if col == 0 else "X" for col in row]) for row in board])


# Add a piece to the board at the given position, and return a new board (doesn't change original)
def add_piece(board, row, col):
    return board[0:row] + [board[row][0:col] + [1, ] + board[row][col + 1:]] + board[row + 1:]


def successors4(board):
    res = []
    flag = False
    # if count_pieces(board) < N:
    for r in range(0, N):
        if count_on_row(board, r) == 0:
            for c in range(0, N):
                if count_on_col(board, c) == 0: #and not (r, c) == (X_r, X_c):
                    if type_name in ['nqueen', 'nqueens']:
                        if count_on_diag(board, r, c):
                            succ = add_piece(board, r, c)
                            res.append(succ)

                    else:
                        res.append(add_piece(board, r, c))
            flag = True

        if flag:
            break

    # print(res)
    return res


# Get list of successors of given board state
def successors(board):
    return [add_piece(board, r, c) for r in range(0, N) for c in range(0, N)]


# check if board is a goal state
def is_goal(board):
    return count_pieces(board) == N  and board[X_r][X_c] == 0 # and \
    # all( [ count_on_row(board, r) <= 1 for r in range(0, N) ] ) and \
    # all( [ count_on_col(board, c) <= 1 for c in range(0, N) ] ) #and not board[R][C]==1


# Solve n-rooks!
def solve(initial_board):
    if type_name in ['nqueen', 'nrook', 'nrooks', 'nqueens']:
        fringe = [initial_board]
        de_fringe = []
        while len(fringe) > 0:
            temp = fringe.pop()
            de_fringe.append(temp)
            for s in successors4(temp):
                rot_s = numpy.rot90(numpy.asarray(s)).tolist()
                if s not in fringe and numpy.fliplr(numpy.asarray(s)).tolist() not in fringe and \
                               numpy.flipud(numpy.asarray(s)).tolist() not in fringe and rot_s not in fringe and \
                        numpy.rot90(numpy.asarray(rot_s)).tolist() not in fringe:
                    if is_goal(s):
                        return (s)
                    elif count_pieces(s) == N:
                        de_fringe.append(s)
                    else:
                        fringe.append(s)

        return False
    else:
        print "Wrong type."
        return False


# This is N, the size of the board. It is passed through command line arguments.
type_name = sys.argv[1]
N = int(sys.argv[2])
X_r = int(sys.argv[3]) - 1
X_c = int(sys.argv[4]) - 1

# The board is stored as a list-of-lists. Each inner list is a row of the board.
# A zero in a given square indicates no piece, and a 1 indicates a piece.
initial_board = [[0] * N] * N
print ("Starting from initial board:\n" + printable_board(initial_board) + "\n\nLooking for solution...\n")
solution = solve(initial_board)
solution[X_r][X_c]=-1
print (printable_board(solution) if solution else "Sorry, no solution found. :(")
