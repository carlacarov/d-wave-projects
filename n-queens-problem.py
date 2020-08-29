#The n-queens problem consists on placing n queens in an nxn chessboard without killing each other

import numpy as np
from itertools import combinations
import dimod
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import neal

#Set the problem as a QUBO: 
#Solution: a sequence of squares that represents the board with queens in it
#Binary variables: indicate whether a square has a queen (1) or not (0)
#Objective function: maximize the number of queens placed
#Constraints: only one queen per row, per column and per diagonal

#Create a class of the chessboard
class Chessboard:
    #Take input boardsize
    def __init__(self, boardsize):
        self.boardsize = boardsize
       
        #Create a matrix of size nxn, the board itself
        self.boardarray = np.array([[0]*self.boardsize]*self.boardsize)
        #Put alternating 0s and 1s, which will correspond to the colours of the chessboard in its plot
        for row in range(0,len(self.boardarray)):
            for col in range(row%2,len(self.boardarray),2):
                self.boardarray[row][col] = 1

        #Put name of the squares in the board as "cmrn", m being the column number and n being the row number
        self.numbered_board = numbered_board = []
        for row in range(0,len(self.boardarray)):
            for col in range(0,len(self.boardarray)):
                numbered_board.append("c{}r{}".format(col,row))

        #Divide the board in rows, creating a list with nested lists for each row
        self.numbered_rows = numbered_rows = []
        for row in range(0,self.boardsize):
            numbered_rows.append(self.numbered_board[self.boardsize*row:((row+1)*self.boardsize)])
    
        #Divide the board in columns, creating a list with nested lists for each column
        self.numbered_cols = numbered_cols = []
        for col in range(0,self.boardsize):
            sq = col
            numbered_cols.append([])
            while sq <= len(self.numbered_board)-1:
                numbered_cols[col].append(self.numbered_board[sq])
                sq+=self.boardsize

        #Divide the board in diagonals, creating a list with nested lists for each diagonal
        self.numbered_diags = numbered_diags = []
        numbered_array = np.array(self.numbered_rows) #Convert board list into an array
        #Add the diagonals, i.e. [[c0r1,c1r2],[c0r0,c1r1,c2r2],[c1r0,c2r1]]
        for diag in range (-len(numbered_array)+2,len(numbered_array)-1):
            numbered_diags.append(list(numbered_array.diagonal(diag)))
        #Add the antidiagonals, i.e. [[c0r1,c1r0],[c0r2,c1r1,c2r0],[c1r2,c2r1]]
        for antidiag in range (-len(numbered_array)+2,len(numbered_array)-1):
            numbered_diags.append(list(np.flipud(numbered_array).diagonal(antidiag)))

    #Represent where's a queen by putting a 2
    def set_queen (self,row,col):
        self.boardarray[row][col] = 2
    
    #Check if the solution obtained is valid
    def check_solution (self):
        filter_arr = self.boardarray > 1
        #Check that there's only 1 queen per diagonal
        for sq in range(-(len(self.boardarray))+2, len(self.boardarray)-1):
            if filter_arr.diagonal(sq).sum() > 1:
                return False
        for sq in range(-(len(self.boardarray))+2, len(self.boardarray)-1):
            if np.flipud(filter_arr).diagonal(sq).sum() > 1:
                return False
        return True
        #Count the non-zero elements
        queenspercol = np.count_nonzero(filter_arr, axis=0)
        queensperrow = np.count_nonzero(filter_arr, axis=1)
        #Check that there's only 1 queen per column
        invalid_cols = queenspercol > 1
        if invalid_cols.sum() > 0:
            return False
        #Check that there's only 1 queen per row
        invalid_rows = queensperrow > 1
        if invalid_rows.sum() > 0:
            return False

    def plot (self):
        #Plot the values of a 2D matrix or array as colour-coded image
        cmap = ListedColormap(['k', 'w', 'b']) #black, white, blue
        fig, ax = plt.subplots()
        ax.matshow(self.boardarray,cmap=cmap)
        plt.show()

def coefficients ():
    linear = {}
    quadratic = {}

    #Add linear coefficients to all the squares in the board
    for sq in range(0,len(b1.numbered_board)):
        linear[b1.numbered_board[sq]] = -1

    #Add quadratic coefficients to the pairs of squares in the same row
    for row in range(0,len(b1.numbered_rows)):
        for rowpair in list(combinations(b1.numbered_rows[row],2)):
            quadratic[rowpair] = 2
    #Add quadratic coefficients to the pairs of squares in the same column
    for col in range(0,len(b1.numbered_cols)):
        for colpair in list(combinations(b1.numbered_cols[col],2)):
            quadratic[colpair] = 2
    #Add quadratic coefficients to the pairs of squares in the same diagonal
    for diag in range(0,len(b1.numbered_diags)):
        for diagpair in list(combinations(b1.numbered_diags[diag],2)):
            quadratic[diagpair] = 2
    
    return linear, quadratic

#Lowest energy solution, where there's 1 queen per row and 1 per column
def choosevar (linear, quadratic):
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0, 'BINARY')
    sampler = neal.SimulatedAnnealingSampler()
    #100 samples gets several valid solutions
    num_iter = int(100)
    sampleset = sampler.sample(bqm, num_reads=num_iter)
    sampleset_iterator = sampleset.samples(num_iter)
    print(sampleset)

    #From the samples obtained, check if the solution is valid and plot it in a board
    for sample in sampleset_iterator:
        for el in sample:
            if sample.get(el)==1:
                row = int(el[el.find("r")+1:])
                col = int(el[1:el.find("r")])
                b1.set_queen(row,col)
        if b1.check_solution():
            b1.plot()


#Instance an object of the class
s=5
b1 = Chessboard(s)
linear, quadratic = coefficients()
choosevar(linear,quadratic)