import dimod
import neal
import numpy as np
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

N = 10
d = 2
k = 3

p = []
for q in range(0,k):
    p.append(2**q)

x, y = make_blobs(n_samples=N, n_features=d, centers=2, center_box=(-10.0,10.0))
for target in range(0,N):
    if y[target]==0:
        y[target] = -1

plt.scatter(x[:, 0], x[:, 1], marker='o', c=y, s=25, edgecolor='k')

#classic
model = SVC(kernel="linear", C=15)
model.fit(x, y)

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
for w in range(0,d):
    for q in range(0,k):
        linear["w{}(q{})".format(d,q)] = -(p[q]+p[q]**2)
for q in range(0,k):
    linear["b(q{})".format(q)] = -p[q]
for a in range (0,N):
    for q in range(0,k):
        linear["a{}(q{})".format(a,q)] = -p[q]

quadratic = {}
for w in range(0,d):
    for q in it.combinations(np.arange(0,k),2):
        quadratic[("w{}(q{})".format(w,q[0]),"w{}(q{})".format(w,q[1]))] = -2*p[q[0]]*p[q[1]]
for a in range(0,N):
    for w in range(0,d):
        for q in it.combinations_with_replacement(np.arange(0,k),2):
            quadratic[("a{}(q{})".format(a,q[0]),"w{}(q{})".format(w,q[1]))] = x[a][w]*y[a]*p[q[0]]*p[q[1]]
for a in range(0,N):
    for q in it.combinations_with_replacement(np.arange(0,k),2):
        quadratic[("a{}(q{})".format(a,q[0]),"b(q{})".format(q[1]))] = y[a]*p[q[0]]*p[q[1]]

bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0, 'BINARY')
print(bqm)
sampler = neal.SimulatedAnnealingSampler()
num_iter = int(100)
sampleset = sampler.sample(bqm, num_reads=num_iter)
sampleset_iterator = sampleset.samples(num_iter)
print(sampleset)

classic_weights = model.coef_
print(classic_weights)

weights = {}
for w in range(0,d):
    for q in range(0,k):
        weights["w{}".format(w)] =+ sampleset_iterator[0]["w{}(q{})".format(w,q)]*p[q]
print(weights)

classic_intercept = model.intercept_
print(classic_intercept)

b = {}
for q in range(0,k):
    b["b"] =+ sampleset_iterator[0]["b(q{})".format(q)]*p[q]
print(b)

classic_lagrange_multipliers = np.abs(model.dual_coef_)
print(classic_lagrange_multipliers)

sv_classic_lagrange_nums = []
for v in range(0,len(sv)):
    for data in range(0,N):
        if sv[v][0]==x[data][0] and sv[v][1]==x[data][1]:
            sv_classic_lagrange_nums.append("a{}".format(data))
sv_classic_lagrange_nums.sort()
print(sv_classic_lagrange_nums)

lagrange_multipliers = {}
for a in range(0,N):
    for q in range(0,k):
        lagrange_multipliers["a{}".format(a)] =+ sampleset_iterator[0]["a{}(q{})".format(a,q)]*p[q]
print(lagrange_multipliers)

sv_lagrange_nums = []
for a in range(0,N):
    if lagrange_multipliers["a{}".format(a)]==max(lagrange_multipliers.values()):
        sv_lagrange_nums.append("a{}".format(a))
sv_lagrange_nums.sort()
print(sv_lagrange_nums)