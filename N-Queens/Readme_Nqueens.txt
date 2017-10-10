This file, a0.py contains the code which is a solution to the following question:

Now, create a new program, a0.py , with several additional features, including the ability to solve both
the N-rooks and N-queens problems, and the ability to solve these problems when when exactly one
position on the board is unavailable (meaning that you cannot place a piece there). Your program must
accept four arguments. The rst one is either nrookÿ or nqueen , which signies the problem type. The
second is the number of rooks or queens (i.e., N). The last two arguments represent the coordinate of
the unavailable position. For example, 2 3ÿ means the square at the second row and the third column
is unavailable. Assuming a coordinate system where (1,1) is at the top-left of the board. Taking the
7-rooks problem with position (1, 1) unavailable as an example, the format and one possible result
could be:
[<>djcran@tank ~]$ python a0.py nrook 7 1 1
X R _ _ _ _ _
R _ _ _ _ _ _
_ _ R _ _ _ _
_ _ _ R _ _ _
_ _ _ _ R _ _
_ _ _ _ _ R _
_ _ _ _ _ _ R
ÿwhere Rÿ indicates the position of a rook, underscore marks an empty square, and Xÿ shows the unavailable
position. Or for the 8-queens problem with (1, 2) unavailable, one possible result could be:
[<>djcran@tank ~]$ python a0.py nqueen 8 1 2
_ X _ _ Q _ _ _
_ _ _ _ _ _ Q _
_ Q _ _ _ _ _ _
_ _ _ _ _ Q _ _
_ _ Q _ _ _ _ _
Q _ _ _ _ _ _ _
_ _ _ Q _ _ _ _
_ _ _ _ _ _ _ Q
ÿwhere the Q 's indicate the positions of queens on the board. As a special case, the coordinate 0 0
ÿindicates that there are no unavailable squares. Please print only the solution in exactly the above
format and nothing else . The output format is important because we will use an auto-grading script
to test and grade your code.

