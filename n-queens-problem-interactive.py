#The n-queens problem consists on placing n queens in an nxn chessboard without killing each other

import numpy as np
from itertools import combinations
import dimod
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import neal
import streamlit as st
import pandas as pd

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

        self.placed_queens=[]

    def get_square_list(self):
        square_list=[]
        for row in self.numbered_board:
            square_list.append(row)
        return square_list

    #Represent where's a queen by putting a 2
    def set_queen (self,row,col):
        self.boardarray[row][col] = 2
        self.placed_queens.append("c{}r{}".format(col,row))

    def set_queens_by_name (self,squares):
        for i in range(0,len(squares)):
            square=squares.iloc[i]['square']
            row = int(square[square.find("r")+1:])
            col = int(square[1:square.find("r")])
            self.set_queen(row,col)
    
    def count_queens(self):
        count = 0
        for row in range(0,len(self.boardarray)):
            for col in range(0,len(self.boardarray)):
                if self.boardarray[row][col]==2:
                    count+=1
        return count

    def get_col_row(self, square_name):
        row = int(square_name[square_name.find("r")+1:])
        col = int(square_name[1:square_name.find("r")])
        return col,row

    def is_in_same_row(self, square_name1,square_name2):
        col1,row1=self.get_col_row(square_name1)
        col2,row2=self.get_col_row(square_name2)
        return row1==row2

    def is_in_same_col(self, square_name1,square_name2):
        col1,row1=self.get_col_row(square_name1)
        col2,row2=self.get_col_row(square_name2)
        return col1==col2

    def is_in_same_diag(self, square_name1,square_name2):
        col1,row1=self.get_col_row(square_name1)
        col2,row2=self.get_col_row(square_name2)
        return np.abs(row1-row2)==np.abs(col1-col2)

    def count_dead_queens(self):
        count_dead = 0
        for i in range(0,len(self.placed_queens)):
            for j in range(i+1,len(self.placed_queens)):
                if self.is_in_same_col(self.placed_queens[i],self.placed_queens[j]) or self.is_in_same_row(self.placed_queens[i],self.placed_queens[j]) or self.is_in_same_diag(self.placed_queens[i],self.placed_queens[j]):
                    count_dead+=1
        return count_dead

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

    def get_plot (self):
        #Plot the values of a 2D matrix or array as colour-coded image
        if len(self.placed_queens)==0:
            cmap = ListedColormap(['w', 'k'])
        else:
            cmap = ListedColormap(['w', 'k', 'b']) #white, black, blue
        fig, ax = plt.subplots()
        ax.matshow(self.boardarray,cmap=cmap)
        return plt

def coefficients (board):
    linear = {}
    quadratic = {}

    #Add linear coefficients to all the squares in the board
    for sq in range(0,len(board.numbered_board)):
        linear[board.numbered_board[sq]] = -1

    #Add quadratic coefficients to the pairs of squares in the same row
    for row in range(0,len(board.numbered_rows)):
        for rowpair in list(combinations(board.numbered_rows[row],2)):
            quadratic[rowpair] = 2
    #Add quadratic coefficients to the pairs of squares in the same column
    for col in range(0,len(board.numbered_cols)):
        for colpair in list(combinations(board.numbered_cols[col],2)):
            quadratic[colpair] = 2
    #Add quadratic coefficients to the pairs of squares in the same diagonal
    for diag in range(0,len(board.numbered_diags)):
        for diagpair in list(combinations(board.numbered_diags[diag],2)):
            quadratic[diagpair] = 2
    
    return linear, quadratic

#Lowest energy solution, where there's 1 queen per row and 1 per column
def choosevar (linear, quadratic, board):
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
                board.set_queen(row,col)
        if board.check_solution():
            return board.get_plot()
            

st.title('N-queens problem')

st.header('Description of the problem')
st.text('The n-queens problem consists on placing n queens in an nxn chessboard without killing')
st.text(' each other. You can try to solve the problem by yourself by writing the names of squares')
st.text(' and clicking "Add" or "Remove" to place or delete a queen in that square.')

st.subheader('Objective function')
st.text('The objective function is a function which tells you the energy of your system.')
st.latex(r'''
E = \sum_{i} a_i q_i + \sum_{i,j} b_{i,j} q_i q_j
''')
st.text('The aim of quantum annealing is to minimise this function. If you express your problem')
st.text(' with binary variables and set properly linear and quadratic coefficients, a and b, ')
st.text(' the solution to the problem encoded, which is a set of 0s and 1s, will correspond to a')
st.text(' low energy configuration.')

show_sol=st.sidebar.checkbox('Solve with quantum annealing')

s=st.sidebar.slider('Board size', 4, 10, 8)

#Instance an object of the class for the user
board = Chessboard(s)
square_list = board.get_square_list()

placed_queens = pd.read_json('selected_squares.json')
if not placed_queens.empty:
    placed_queens.set_index('square')

square_name=st.sidebar.text_input('Square name (example c0r0,c3r4,...)')
if st.sidebar.button('Add'):
    square={'square':square_name}
    placed_queens=placed_queens.append(square,ignore_index=True)
    placed_queens.to_json('selected_squares.json')

if st.sidebar.button('Remove'):
    placed_queens.drop(placed_queens[(placed_queens.square==square_name)].index, inplace=True)
    placed_queens.to_json('selected_squares.json')

st.write('You selected:', placed_queens)
board.set_queens_by_name(placed_queens)
energy = board.count_queens()*(-1) + board.count_dead_queens()*2
st.write('Your energy:', energy)
energy_sol = s*(-1)
st.write('Energy of the solution:', energy_sol)
st.pyplot(board.get_plot())

st.header('Quantum annealing')
st.text('Quantum annealing is a quantum computational method in which we take advantage of the')
st.text(' fundamental properties of quantum mechanics. It consists on the evolution of a system of')
st.text(' qubits, which are the quantum bits. Qubits have the property of quantum superposition, ')
st.text(' which means that they can be in the 0 state and the 1 state at the same time.')
st.text('At the beginning of the anneal, all the qubits are in such superposition of states.')
st.text('Also, qubits have another interesting quantum property called entanglement, which is an')
st.text(' invisible connection between pairs of two qubits. At the beginning of the anneal, you also')
st.text(' set the pairs of entangled qubits. In the end, you have a system of superposed qubits')
st.text(' entangled with each other. Together they define what is called an energy landscape.')
st.text('The aim of quantum annealing is to minimise the energy landscape of the system. To do so,')
st.text(' it goes through a process of annealing to reach the configuration of qubits that has')
st.text(' the lowest energy. This final configuration no longer has superposed entangled qubits')
st.text(' but it is rather a set of qubits with deterministic states, they are either 0 or 1.')
st.text('The interesting fact is that the set of 0s and 1s that you have at the end of the anneal')
st.text(' represents the solution to your problem.')


#Instance an object of the class for the annealer
if show_sol:
    board_solution = Chessboard(s)
    linear, quadratic = coefficients(board_solution)
    st.write('Solution found with quantum annealing:')
    st.pyplot(choosevar(linear,quadratic,board_solution))
    st.text('Thanks to quantum annealing, all the possible configurations of 0s and 1s are explored at')
    st.text(' the same time and the system collapsed to that configuration of the lowest energy, ')
    st.text(' which is the solution to the problem!')