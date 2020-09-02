# d-wave-projects
## Project 1: AND gate
## Project 2: Select 1 variable of many
## Project 3: Select 2 variables of many
## Project 4: Knight's tour problem
## Project 5: N-queens problem
## Project 6: N-queens problem interactive
## Project 7: Travelling salesman problem
## Project 8: Travelling salesman problem interactive
## Project 9: Quantum support vector machine (SVM)

# Basic concepts
Quantum annealing requires the problem to be expressed as a BQM (Binary Quadratic Model).
The BQM represents the problem as an objective function with binary variables and linear and quadratic coefficients.
The aim of the annealing process is to find low-energy states, which are good solutions for your problem.

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
