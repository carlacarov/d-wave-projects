#The knights tour problem consists on visiting every square of the chess board just once
#doing valid knight movements

import dimod
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from collections import defaultdict
import itertools
import neal

#Set the problem as a QUBO: 
#Solution: a sequence of squares that represents the path
#Identify the order in the sequence of movements with a number 1,2,...,n
#Identify the squares with numrow and numcol
#Binary variables: indicate whether a square is visited in the nth position of the path
#Objective function: maximize the number of squares visited
#Constraints: only valid knight movements, max one visit per square, only one square in the nth movement

#I got the idea from https://github.com/dwave-examples/maze

#Helper function returning True if the movement is valid, returns False otherwise
def valid_move(square1,square2):
    #Transform numerical square to row and col
    row1=square1 // board_size
    col1=square1 % board_size
    row2=square2 // board_size
    col2=square2 % board_size
    #Valid movement if the difference of rows is 2 and difference of cols is 1 or viceversa
    if ((abs(row2-row1)==2 and abs(col2-col1)==1) or (abs(row2-row1)==1 and abs(col2-col1)==2)):
        return True
    else:
        return False

def get_movement_label(square,pos):
    #Returns a string representing the square visited in posth of the path 
    return "{}in{}".format(square,pos)

#Validates solution
def valid_solution(solution):
    num_squares_visited=0
    squares_visited=set() #Unordered set
    prev_square=-1 #Initialise to mark the initial position
    valid_path=True
    for pos in range(number_of_squares):
        for sq in range(number_of_squares):
            if (solution[get_movement_label(sq,pos)]==1):
                #sq is visited in pos
                num_squares_visited +=1
                if (prev_square>=0): #There's a previous square
                    if valid_move(prev_square,sq):
                        if sq not in squares_visited:
                            print("valid  move at pos {} sq {} - {}".format(pos,prev_square,sq))
                            squares_visited.add(sq)
                        else:
                            print("square visited twice")
                            valid_path=False    
                    else:
                        print("invalid move at pos {} from {} to {}".format(pos,prev_square,sq))
                        valid_path=False
                prev_square=sq
    print("num_squares_visited {}".format(num_squares_visited))
    
    if (num_squares_visited==number_of_squares and valid_path):
        return True #Path is valid so far and all the squares have been visited
    else:
        return False

def create_QUBO():
    #Creating the QUBO
    Q = defaultdict(int)
    penalty=2

    #Constraint one square in just one position
    for sq in range(number_of_squares):
        for pos_1 in range(number_of_squares):
            #Add negative value to the favour just one variable selected, 
            #otherwise the optimal solution would be not to move
            #These are the quadratic coefficients of the QUBO matrix
            Q[(get_movement_label(sq, pos_1), get_movement_label(sq, pos_1))] -= 1
            #If sq is visited in pos_1, I have to penalise its visit in any other position
            for pos_2 in range(pos_1+1, number_of_squares):
                Q[(get_movement_label(sq, pos_1), get_movement_label(sq, pos_2))] += penalty

    #Constraint that each position has exactly one square
    for pos in range(number_of_squares):
        for sq in range(number_of_squares):
            Q[(get_movement_label(sq, pos), get_movement_label(sq, pos))] -= 1
            for sq2 in range(sq+1,number_of_squares):
                Q[(get_movement_label(sq, pos), get_movement_label(sq2, pos))] += penalty

    #Constraint that penalises invalid movements
    #Get all the combinations of movements combined with all the pairs of consecutive positions
    for x, y in itertools.combinations(range(number_of_squares), 2):
        for pos in range(number_of_squares-1):
            nextpos = pos + 1
            if (not valid_move(x,y)):
                Q[(get_movement_label(x, pos), get_movement_label(y, nextpos))] += 2*penalty
                Q[(get_movement_label(y, pos), get_movement_label(x, nextpos))] += 2*penalty
    return Q

def initialize_board(board_size=4):
    board = np.zeros(board_size*board_size)
    #Reshape things into a nrows grid.
    nrows, ncols = board_size,board_size
    board = board.reshape((nrows, ncols))
    #Set black squares
    for r in range(0, nrows,2):
        board[r][1::2]=1
    for r in range(1, nrows,2):
        board[r][0::2]=1
    return board

#I took this routine of matplotlib from a sample to draw a board with the solution
def show_solution_board(solution):      
    board= initialize_board(board_size)
    cmap = ListedColormap(['k', 'w', 'g']) #black, white, green
    fig, ax = plt.subplots()
    ax.matshow(board,cmap=cmap)
    for el in solution:
        if (solution[el]==1):
            sq_pos=el.split("in")
            square_number=sq_pos[0]
            square_position=sq_pos[1]
            row =int(square_number) // board_size
            col=int(square_number) % board_size
            print ("row: {} col {}".format(row,col))
            ax.text(row, col, square_position, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    plt.show()
    
board_size=5
number_of_squares=board_size*board_size
bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
Q = create_QUBO()

print("qubo is:{}".format(Q))

sampler = neal.SimulatedAnnealingSampler()

#100 samples always gets several valid solutions
response = sampler.sample_qubo(Q, num_reads=100)

#The sampler returns low energy solutions, but not all of them are valid, so I check it
for solution in response.samples(100):
    if valid_solution(solution):
        print("solution is valid")
        print(solution)
        show_solution_board(solution)
print("end")