import dimod

linear = {'a':-1, 'b':-1, 'c':-1}
quadratic = {('a','b'):2, ('a','c'):2, ('b','c'):2}
bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0, 'BINARY')
sampleset = dimod.ExactSolver().sample(bqm)
print(sampleset)