# d-wave-projects
## Project 1: AND gate
The formulation of an AND gate as a BQM is quite straightforward: $E(x_1,x_2,y_1) = 2y_1 + x_1x_2 - \frac{3}{2}x_1y_1 - \frac{3}{2}x_2y_1$ (up to a global factor), where $x_1,x_2$ are the inputs and $y_1$ is the output of the AND gate. This formulation corresponds to file and-gate-bqm.py. Alternatively, D-Wave provides a dwavebinarycsp library, which represents logic gates such as AND, OR, NOT, etc. as binary constraint satisfaction problems (CSP). Indeed, an AND gate can be viewed as the constraint $y_1 = x_1x_2$, as implemented in and-gate-csp.py.
## Project 2: Select 1 variable of many
This is a simple problem that selects one variable out of many, which can be formulated as BQM as $E(a,b,c)  = -(a+b+c) + 2(ab+ac+bc)$. This has been implemented using D-Wave's exact solver (in select-one.py) and embedding the problem in the annealer's architecture (in select-one-dwave-solver.py).
## Project 3: Select 2 variables of many
This is a very similar problem, consisting of selecting 2 variables out of many, which is formulated as $E(a,b,c) = -3(a+b+c) + 2(ab+ac+bc)$, implemented in select-two.py.
## Project 4: Knight's tour problem
The knight's tour problem involves visiting every square of a chess board just once and performing valid knight movements. This problem has been formulated as a QUBO problem by identifying the order in the sequence of movements with the numbers 1, 2,...,n, and binary variables indicating whether a square is visited in the nth position of the path. Then, the constraints impose that only valid knight movements are performed, each square is visited at most once, and only one square is visited in the nth movement. Details of the formulation and implementation can be found at the-knights-tour-pos.py.
## Project 5: N-queens problem
The n-queens problem consists of placing n queens in an nxn chessboard without killing each other. This problem has been formulated as a QUBO problem with binary variables indicating whether a square has a queen (1) or not (0) and imposing as constraints that there are not more than one queen per row, column or diagonal. Details of the formulation and implementation can be found at n-queens-problem.py. Besides, you can also read a pedagogical explanation here: https://medium.com/@carlacarov/solving-the-n-queens-problem-with-quantum-annealing-b13ee199b210.
## Project 6: N-queens problem interactive
Using the framework Streamlit, I have also programmed an interactive version of the n-queens problem, in which a chessboard is displayed, and queens can be manually placed, while seeing how the energy of the system varies. This provides a pedagogical and intuitive view of the QUBO formulation of this problem. In addition, it includes a button that solves the problem using quantum annealing and plots the solution on the board. The application can be run from the cmd by typing "streamlit run n-queens-problem-interactive.py".
## Project 7: Travelling salesman problem
The travelling salesman problem (TSP) is a well-known graph theory problem that consists of finding the Hamiltonian cycle such that the sum of the weights of each edge in the cycle is minimized, in other words, finding the shortest route that visits every node exactly once. This problem is solved using the D-Wave's SDK high-level function dnx.traveling_salesperson (see travelling-salesman.py). Behind the scenes it creates a QUBO formulation where each binary variable labelled c, t represents the city c visited in step t. Then it sets quadratic coefficients to minimise distance subject to the constraint: each city is visited exactly once.
## Project 8: Travelling salesman problem interactive
Again, using the framework Streamlit, I have programmed an interactive and pedagogical version of the TSP, in which a map is displayed, and the user is allowed to choose the initial city of the journey and, then, calculate the optimal route with a simulated annealing sampler, which is plotted. The cities and distances between them can be manually added to the cities.json and distances.json files. To execute, type "streamlit run travelling-salesman-interactive.py" in your cmd.
## Project 9: Quantum support vector machine (qSVM)
A support vector machine (SVM) is a supervised machine learning algorithm, mainly used for classification tasks, that finds the optimal hyperplane separating classes of data. This algorithm can be formulated as a QUBO problem, using a proper binary-to-decimal encoding, from the Lagrangian optimisation problem $\underset{a}{min} L(a) = \frac{1}{2}\sum_{i,j}^{N-1}\sum_{p,q}^{K-1} 2^{p+q} a_p^{(i)}a_q^{(j)} y_i y_j k(x_i,x_j) - \sum_i^{N-1}\sum_p^{K-1} 2^p a_p^{(i)}$, where $a_p^{(i)}$ are binary variables used in the decimal encoding (using powers of 2, $2^p$) of Lagrange multipliers ($\alpha_i = \sum_p^{K-1} 2^p a_p^{(i)}$), $x_i,x_j$ are the feature vectors, whose "similarity" is computed with a chosen kernel $k(x_i,x_j)$, and $y_i,y_j$ are the classes of each vector. This minimisation problem is subject to $\sum_i^{N-1} \alpha_i y_i = 0$, which I chose not to impose as a constraint in the QUBO formulation, but rather opt for sampling over the obtained annealing solutions. In quantum-svm-toy.py, the qSVM is formulated and executed for a simple toy model, this is, a small synthetic dataset, to gain some intuition and easily compare it to the classical SVM. One can refer directly to quantum-svm.py for a full implementation of the qSVM using D-Wave's solver and simulated annealing for a real dataset, so you can find the details of the formulation, alongside the computation of widely-used machine learning metrics (such as accuracy, precision, ROC, AUC, etc.), that allows for the comparison of its performance with that of the classical SVM.

# Basic concepts
Quantum annealing requires the problem to be expressed as a BQM (Binary Quadratic Model).
The BQM represents the problem as an objective function with binary variables and linear and quadratic coefficients.
The annealing process aims to find low-energy states, which are good solutions for your problem.

# Main steps
1. State the problem as a QUBO (Quadratic Unconstrained Binary Optimization)
2. Choose a solver, which is a resource where the minimization process is executed. It can be your local computer, the D-Wave QPU (Quantum Process Unit) or a hybrid approach.
3. Choose a sampler, which is the process to get low-energy states (solutions to your problem).

# Libraries
The main libraries I will be using are the following:
1. dimod: provides a BQM class for both Ising and QUBO problems and it also includes samplers and solvers.
2. neal: simulated annealing sampler.
3. dwave-system: provides classes to solve the problem in the physical D-Wave QPU. It also provides utilities to map automatically your logical BQM into the physical QPU architecture (Minor-Embedding).

D-Wave Ocean software also provides other high-level libraries which help you to formulate a BQM. For instance, if you have a constraint-based problem, you can use dwavebinarycsp. If you have a graph-based problem, then the dwave-networkx may be useful.

Finally, I use some general libraries such as numpy for arrays and matplotlib for plotting, among others.

# Run programs
To execute the programs, you just have to download them and run them with python. For the interactive versions, you have to run them with the framework Streamlit. To download it, write "pip install streamlit" in your cmd and then "streamlit run name_of_program.py" to run the program, which will open a window in your browser.
