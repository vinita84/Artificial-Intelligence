The code in the file assign.py is a solution to the following problem in Python:

The course sta of a certain Computer Science course randomly assigns students to teams for their rst
assignment of the semester to help force people to get used to the real-world situation of working on
diverse groups of teams of programmers. But since they are not completely heartless, the instructors
allow students to give preferences on their teams for future assignments. In particular, each student is
sent an electronic survey and asked to give answers to four questions:
(a) What is your user ID?
(b) Would you prefer to work alone, in a team of two, in a team of three, or do you not have a
preference? Please enter 1, 2, 3, or 0, respectively.
(c) Which student(s) would you prefer to work with? Please list their user IDs separated by commas,
or leave this box empty if you have no preference.
(d) Which students would you prefer not to work with? Please list their IDs, separated by commas.
You can assume every student lls out the survey exactly once. Ideally, student preferences would be
compatible with each other so that the group assignments would make everyone happy, but inevitably
this is not possible because of conicting preferences. So instead, being selsh, the course sta would
like to choose the group assignments that minimize the total amount of work they'll have to do. They
estimate that:
ÿ They need kÿ minutes to grade each assignment, so total grading time is kÿ times number of teams.
ÿ Each student who requested a specic group size and was assigned to a dierent group size will
complain to the instructor after class, taking 1 minute of the instructor's time.
ÿ Each student who is not assigned to someone they requested will send a complaint email, which
will take nÿ minutes for the instructor to read and respond. If a student requested to work with
multiple people, then they will send a separate email for each person they were not assigned to.
ÿ Each student who is assigned to someone they requested notÿ to work with (in question 4 above)
will request a meeting with the instructor to complain, and each meeting will last mÿ minutes. If
a student requested not to work with two specic students and is assigned to a group with both
of them, then they will request 2 meetings.
The total time spent by the course sta is equal to the sum of these components. Your goal is to write
a program to nd an assignment of students to teams that minimizes the total amount of work the
course sta needs to do, subject to the constraint that no team may have more than 3 students. Your
program should take as input a text le that contains each student's response to these questions on a
single line, separated by spaces. For example, a sample le might look like:
djcran 3 zehzhang,chen464 kapadia
chen464 1 _ _
fan6 0 chen464 djcran
zehzhang 1 _ kapadia
kapadia 3 zehzhang,fan6 djcran
steflee 0 _ _
ÿwhere the underscore character ( ) indicates an empty value. Your program should be run like this:
./assign.py [input-file] [k] [m] [n]
ÿwhere k , m , and nÿ are values for the parameters mentioned above, and the output of your program
should be a list of group assignments, one group per line, with user names separated by spaces followed
by the total time requirement for the instructors, e.g.:
djcran chen464
kapadia zehzhang fan6
steflee
534

