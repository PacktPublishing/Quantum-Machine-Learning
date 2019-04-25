
# Example 02 - Unsupervised Learning - Clustering

# Quantum Approximate Optimization Algorithm (QAOA) - Proof of Concept


***
#### Thank you to these original authors for making the template Jupyter notebook available to the machine learning community.
Peter Wittek and team from the Quantum Machine Learning edX inaugural course. This course will be offered again soon. Check out https://www.edx.org/.

#### References and additional details:
https://gitlab.com/qosf/qml-mooc/blob/master/qiskit_version/10_Discrete_Optimization_and_Unsupervised_Learning.ipynb

#### Quantum Computing Framework
- <sup>[1]</sup>IBM QISKit


# Running the Code

Installing everything you need on your own laptop takes a little time, so for right now the easiest way to get started is to use a "Binder image".  This setup lets you use the code notebooks via the web IMMEDIATELY. I think this is your best approach, for now, because it means you can run the code today to get a feel for how this example of a Quantum Machine Learning algorithm works. In the future, you can follow the full installation process (2 hours) and run your own code. Let's get started. 

# Overview

1) Click on [the Binder image](https://mybinder.org/v2/gh/PacktPublishing/Quantum-Machine-Learning/blob/master?urlpath=https%3A%2F%2Fgithub.com%2FPacktPublishing%2FQuantum-Machine-Learning%2Fblob%2Fmaster%2FQAOA.ipynb) (2 min)

2) Run the Jupyter Notebook. If this is your first time using a Jupyter Notebook, select a cell (it will now have a blue border).  Now SHIFT then RETURN to run the code on the remote server.  The notes below will add explanations and a lot more context to the published notebooks. I thought this was a much more efficient way to help you because you can always link to the newest published code, download it, and use these notes to modify as you wish. 

### Suggestion:  Run the code _as is_ the first time.  You can then go back and make changes to see how the settings work.


# Code Block 01

In order to cluster some data points, we need to know which ones are close to each other.  Let's generate a dataset and assign 2 labels (classes), one red and one green. Clustering very high dimensional data (lots of variables) is a difficult problem for conventional computers, so there is some expectations that quantum machine learning will have some advantage here.  This toy problem will give you an example using the QAOA algorithm to find the best cluster.


```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

n_instances = 10
class_1 = np.random.rand(n_instances//2, 3)/5
class_2 = (0.6, 0.1, 0.05) + np.random.rand(n_instances//2, 3)/5
data = np.concatenate((class_1, class_2))
colors = ["red"] * (n_instances//2) + ["green"] * (n_instances//2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', xticks=[], yticks=[], zticks=[])
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors)
```

# Code Block 02
We are going to use a measure for distance called the Euclidean distance. (There are actually several distance measures (https://en.wikipedia.org/wiki/Norm_(mathematics)).  We can calculate all the pairwise distances between the points.


```python
import itertools
w = np.zeros((n_instances, n_instances))
for i, j in itertools.product(*[range(n_instances)]*2):
    w[i, j] = np.linalg.norm(data[i]-data[j])
```

# Code Block 03


### Using QAOA to solve to find the best way to cluster the points (optimization)

We are going to solve the clustering problem using the QAOA algorithm as the optimization part to get the best answer for how to cluster these points.  If we were to use the distance information between the points, we could , draw a graph.  The Max Cut approach asked the question, "How can I cut this graph into 2 parts?"  The max would be the maximal distances that cross the cut.

We'll make use of the built in libraries from QISKit.


```python
from qiskit_aqua import get_aer_backend, QuantumInstance
from qiskit_aqua.algorithms import QAOA
from qiskit_aqua.components.optimizers import COBYLA
from qiskit_aqua.translators.ising import maxcut
```


```python
qubit_operators, offset = maxcut.get_maxcut_qubitops(w)

#Setting p=1 to initialize the max-cut problem.
p = 1  
optimizer = COBYLA()
qaoa = QAOA(qubit_operators, optimizer, p, operator_mode='matrix')
```

Here the choice of the classical optimizer `COBYLA` was arbitrary. Let us run this and analyze the solution. This can take a while on a classical simulator.

# Code Block 04

This is the output you should expect. The values will be slightly different but this is how it will look.

energy: -0.9232657682915413

maxcut objective: -2.5378918584323404

solution: [0. 0. 1. 1.]

solution objective: 2.9963330796428673



```python
backend = get_aer_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend, shots=100)
result = qaoa.run(quantum_instance)
x = maxcut.sample_most_likely(result['eigvecs'][0])
graph_solution = maxcut.get_graph_solution(x)
print('energy:', result['energy'])
print('maxcut objective:', result['energy'] + offset)
print('solution:', maxcut.get_graph_solution(x))
print('solution objective:', maxcut.maxcut_value(x, w))
```

# References

[1] Otterbach, J. S., Manenti, R., Alidoust, N., Bestwick, A., Block, M., Bloom, B., Caldwell, S., Didier, N., Fried, E. Schuyler, Hong, S., Karalekas, P., Osborn, C. B., Papageorge, A., Peterson, E. C., Prawiroatmodjo, G., Rubin, N., Ryan, Colm A., Scarabelli, D., Scheer, M., Sete, E. A., Sivarajah, P., Smith, Robert S., Staley, A., Tezak, N., Zeng, W. J., Hudson, A., Johnson, Blake R., Reagor, M., Silva, M. P. da, Rigetti, C. (2017). [Unsupervised Machine Learning on a Hybrid Quantum Computer](https://arxiv.org/abs/1712.05771). *arXiv:1712.05771*. <a id='1'></a>
