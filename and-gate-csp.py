import dimod
import dwavebinarycsp
import dwavebinarycsp.factories.constraint.gates as gates
csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)
csp.add_constraint(gates.and_gate(['x1', 'x2', 'y1']))  # add an AND gate
bqm = dwavebinarycsp.stitch(csp)
sampleset = dimod.ExactSolver().sample(bqm)
print(sampleset)