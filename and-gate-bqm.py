import dimod
import neal

linear = {'x1':0, 'x2':0, 'y1':2}
quadratic = {('x1','x2'):1, ('x1','y1'):-1.5, ('x2','y1'):-1.5}
bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0, 'BINARY')
sampleset = dimod.ExactSolver().sample(bqm)
print(sampleset)

#now, solving with simulated annealing
annealingsampler = neal.SimulatedAnnealingSampler()
sampleset2 = annealingsampler.sample(bqm,num_reads=3)
for solution in sampleset2:
        print ("low energy sample: {}".format(solution))