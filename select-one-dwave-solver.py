import dimod
import dwave.system.samplers as dw
from dwave.system import EmbeddingComposite

linear = {'a':-1, 'b':-1, 'c':-1}
quadratic = {('a','b'):2, ('a','c'):2, ('b','c'):2}
bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0, 'BINARY')

dwave_sampler = EmbeddingComposite(dw.DWaveSampler())
sampleset = dwave_sampler.sample(bqm,num_reads=100)
print(sampleset)