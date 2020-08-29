import dimod
import neal
import numpy as np
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC

x = np.array([[-6,2],[-2,2],[2,2],[5,2]])
y = np.array([1,1,-1,-1])

plt.scatter(x[:, 0], x[:, 1], marker='o', c=y, s=25, edgecolor='k')

model = SVC(kernel="linear", C=2)
model.fit(x,y)

#plot taken from scikit documentation:

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# plot support vectors
sv = model.support_vectors_
ax.scatter(sv[:, 0], sv[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')

plt.show()

linear = {}
linear["a0(q0)"] = -1
linear["a0(q1)"] = -2
linear["a1(q0)"] = -1
linear["a1(q1)"] = -2
linear["a2(q0)"] = -1
linear["a2(q1)"] = -2
linear["a3(q0)"] = -1
linear["a3(q1)"] = -2

def dot_kernel(x_n,x_m):
    dot = x_n[0]*x_m[0] + x_n[1]*x_m[1]
    return dot

quadratic = {}
quadratic[("a0(q0)","a1(q0)")] = y[0]*y[1]*dot_kernel(x[0],x[1])
quadratic[("a0(q0)","a1(q1)")] = 2*y[0]*y[1]*dot_kernel(x[0],x[1])
quadratic[("a0(q1)","a1(q0)")] = 2*y[0]*y[1]*dot_kernel(x[0],x[1])
quadratic[("a0(q1)","a1(q1)")] = 2*2*y[0]*y[1]*dot_kernel(x[0],x[1])
quadratic[("a0(q0)","a2(q0)")] = y[0]*y[2]*dot_kernel(x[0],x[2])
quadratic[("a0(q0)","a2(q1)")] = 2*y[0]*y[2]*dot_kernel(x[0],x[2])
quadratic[("a0(q1)","a2(q0)")] = 2*y[0]*y[2]*dot_kernel(x[0],x[2])
quadratic[("a0(q1)","a2(q1)")] = 2*2*y[0]*y[2]*dot_kernel(x[0],x[2])
quadratic[("a0(q0)","a3(q0)")] = y[0]*y[3]*dot_kernel(x[0],x[3])
quadratic[("a0(q0)","a3(q1)")] = 2*y[0]*y[3]*dot_kernel(x[0],x[3])
quadratic[("a0(q1)","a3(q0)")] = 2*y[0]*y[3]*dot_kernel(x[0],x[3])
quadratic[("a0(q1)","a3(q1)")] = 2*2*y[0]*y[3]*dot_kernel(x[0],x[3])
quadratic[("a1(q0)","a2(q0)")] = y[1]*y[2]*dot_kernel(x[1],x[2])
quadratic[("a1(q0)","a2(q1)")] = 2*y[1]*y[2]*dot_kernel(x[1],x[2])
quadratic[("a1(q1)","a2(q0)")] = 2*y[1]*y[2]*dot_kernel(x[1],x[2])
quadratic[("a1(q1)","a2(q1)")] = 2*2*y[1]*y[2]*dot_kernel(x[1],x[2])
quadratic[("a1(q0)","a3(q0)")] = y[1]*y[3]*dot_kernel(x[1],x[3])
quadratic[("a1(q0)","a3(q1)")] = 2*y[1]*y[3]*dot_kernel(x[1],x[3])
quadratic[("a1(q1)","a3(q0)")] = 2*y[1]*y[3]*dot_kernel(x[1],x[3])
quadratic[("a1(q1)","a3(q1)")] = 2*2*y[1]*y[3]*dot_kernel(x[1],x[3])
quadratic[("a2(q0)","a3(q0)")] = y[2]*y[3]*dot_kernel(x[2],x[3])
quadratic[("a2(q0)","a3(q1)")] = 2*y[2]*y[3]*dot_kernel(x[2],x[3])
quadratic[("a2(q1)","a3(q0)")] = 2*y[2]*y[3]*dot_kernel(x[2],x[3])
quadratic[("a2(q1)","a3(q1)")] = 2*2*y[2]*y[3]*dot_kernel(x[2],x[3])

#option with epsilon that you set:
epsilon = 0
linear["a0(q0)"] =+ epsilon*(y[0]**2)
linear["a0(q1)"] =+ epsilon*(y[0]**2)*4
linear["a1(q0)"] =+ epsilon*(y[1]**2)
linear["a1(q1)"] =+ epsilon*(y[1]**2)*4
linear["a2(q0)"] =+ epsilon*(y[2]**2)
linear["a2(q1)"] =+ epsilon*(y[2]**2)*4
linear["a3(q0)"] =+ epsilon*(y[3]**2)
linear["a3(q1)"] =+ epsilon*(y[3]**2)*4
quadratic[("a0(q0)","a0(q1)")] = epsilon*2*2*(y[0]**2)
quadratic[("a1(q0)","a1(q1)")] = epsilon*2*2*(y[1]**2)
quadratic[("a2(q0)","a2(q1)")] = epsilon*2*2*(y[2]**2)
quadratic[("a3(q0)","a3(q1)")] = epsilon*2*2*(y[3]**2)

bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0, 'BINARY')
sampler = neal.SimulatedAnnealingSampler()
num_iter = int(100)
sampleset = sampler.sample(bqm, num_reads=num_iter)
sampleset_iterator = sampleset.samples(num_iter)
print(sampleset)

classic_lagrange_multipliers = np.abs(model.dual_coef_)
print(classic_lagrange_multipliers)

lagrange_multipliers = {}
lagrange_multipliers["a0"] = sampleset_iterator[0]["a0(q0)"] + sampleset_iterator[0]["a0(q1)"]*2
lagrange_multipliers["a1"] = sampleset_iterator[0]["a1(q0)"] + sampleset_iterator[0]["a1(q1)"]*2
lagrange_multipliers["a2"] = sampleset_iterator[0]["a2(q0)"] + sampleset_iterator[0]["a2(q1)"]*2
lagrange_multipliers["a3"] = sampleset_iterator[0]["a3(q0)"] + sampleset_iterator[0]["a3(q1)"]*2
print(lagrange_multipliers)