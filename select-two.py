import dimod

linear = {'a':-3, 'b':-3, 'c':-3}
quadratic = {('a','b'):2, ('a','c'):2, ('b','c'):2}
bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0, 'BINARY')
sampleset = dimod.ExactSolver().sample(bqm)
print(sampleset)