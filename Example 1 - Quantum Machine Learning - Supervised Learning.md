
# Example 01 - Supervised Learning - Classification

# Quantum Support Vector Machines (qSVM) - Proof of Concept


***
#### Thank you to these original authors and to the IBM Q team, for making the template Jupyter notebook available to the machine learning community.
Vojtech Havlicek<sup>[1]</sup>, Kristan Temme<sup>[1]</sup>, Antonio Córcoles<sup>[1]</sup>, Peng Liu<sup>[1]</sup>, Richard Chen<sup>[1]</sup>, Marco Pistoia<sup>[1]</sup> and Jay Gambetta<sup>[1]</sup>

#### References and additional details:

[1] Vojtech Havlicek, Antonio D. C´orcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, and Jay M. Gambetta1, "Supervised learning with quantum enhanced feature spaces," [arXiv: 1804.11326](https://arxiv.org/pdf/1804.11326.pdf)


#### Quantum Computing Framework
- <sup>[1]</sup>IBM QISKit


# Running the Code

***

Installing everything you need on your own laptop takes a little time, so for right now the easiest way to get started is to use a "Binder image".  This setup lets you use the code notebooks via the web IMMEDIATELY. I think this is your best approach, for now, because it means you can run the code today to get a feel for how this example of a Quantum Machine Learning algorithm works. In the future, you can follow the full installation process (2 hours) and run your own code. Let's get started. 

# Overview

1) Click on [the Binder image](https://mybinder.org/v2/gh/qiskit/qiskit-tutorials/master?filepath=index.ipynb).  (2 min)

2) Scroll down to 1.6 and click on *Artificial Intelligence*.

3) On this next page, select *Quantum SVM for Classification*.

4) Run the Jupyter Notebook. If this is your first time using a Jupyter Notebook, select a cell (it will now have a blue border).  Now SHIFT then RETURN to run the code on the remote server.  The notes below will add explanations and a lot more context to the published notebooks. I thought this was a much more efficient way to help you because you can always link to the newest published code, download it, and use these notes to modify as you wish. 

### Suggestion:  Run the code _as is_ the first time.  You can then go back and make changes to see how the settings work.



# Code Block 01

Run this code block in the Binder instance.  All these code statements are about setting up the tools you need to run the rest of the code.  

- You'll get data from `from qsvm_datasets import *`.
- You'll get many tools from the `qiskit_aqua` libraries.



```python
from qsvm_datasets import *

from qiskit import Aer
from qiskit_aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit_aqua.input import SVMInput
from qiskit_aqua import run_algorithm, QuantumInstance
from qiskit_aqua.algorithms import QSVMKernel
from qiskit_aqua.components.feature_maps import SecondOrderExpansion

# setup aqua logging
import logging
from qiskit_aqua import set_aqua_logging
# set_aqua_logging(logging.DEBUG)  # choose INFO, DEBUG to see the log
```

# Code Block 02 - SKIP 

Come back to this later once you have an account and API token from https://quantumexperience.ng.bluemix.net/qx (5 min)


```python
from qiskit import IBMQ
IBMQ.load_accounts()
```

# Code Block 03 - Configuration

This is where you start configuring the code you want to submit.  Run this block _AS IS_ the first time.

This example uses 2 qubits, `feature_dim=2`

you'll first use a dataset called `ad_hoc_data`.  It is used for training, testing and the finally prediction. There are 20 observations in the training group, and 10 in the test group. You'll train the quantum SVM using those 20 observations. 

You can easily switch to a different dataset, such as the Breast Cancer dataset, by replacing 'ad_hoc_data' to 'Breast_cancer' below.*

After you run the code, you should see 2 plots: 

1) A very complicated pattern.  The goal is to lable these regions using 2 labels: A, B.  
2) A plot of points showing the lable for each color region.


```python
feature_dim=2 # we support feature_dim 2 or 3
sample_Total, training_input, test_input, class_labels = ad_hoc_data(training_size=20, 
                                                                     test_size=10, 
                                                                     n=feature_dim, 
                                                                     gap=0.3, 
                                                                     PLOT_DATA=True)
extra_test_data = sample_ad_hoc_data(sample_Total, 10, n=feature_dim)
datapoints, class_to_label = split_dataset_to_data_and_labels(extra_test_data)
print(class_to_label)
```

# Code Block 04 - Run

This is the block of code that runs the algorithm and gives you the results. 

- You need to set a seed to get the same results if you run this again.  You can change this.
- Notice that the testing output is nearly perfect, %100.  That is because this is a toy example.
- The `ground truth` are the known labels for the data.
- As you can see the algorithm in this particular output predicted the new data (data it had not processed earlier) with a perfect score.  This is very unlikely in real life.  This code is showing that the algorithm works, in principal.
- Notice the block of code that allows you to use the quantum simulator `backend = Aer.get_backend('qasm_simulator')`. Once you have an account, you can choose another system to run your code on. Here is a great article about all the backends available to you: 
https://medium.com/qiskit/qiskit-backends-what-they-are-and-how-to-work-with-them-fb66b3bd0463

#### Expected Output

testing success ratio: 1.0

preduction of datapoints:

ground truth: ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']

prediction:   ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']


```python
seed = 10598

feature_map = SecondOrderExpansion(num_qubits=feature_dim, depth=2, entanglement='linear')
qsvm = QSVMKernel(feature_map, training_input, test_input, datapoints[0])

backend = Aer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024, seed=seed, seed_mapper=seed)

result = qsvm.run(quantum_instance)

"""declarative approach
params = {
    'problem': {'name': 'svm_classification', 'random_seed': 10598},
    'algorithm': {
        'name': 'QSVM.Kernel'
    },
    'backend': {'name': 'qasm_simulator', 'shots': 1024},
    'feature_map': {'name': 'SecondOrderExpansion', 'depth': 2, 'entanglement': 'linear'}
}
algo_input = SVMInput(training_input, test_input, datapoints[0])
result = run_algorithm(params, algo_input)
"""

print("testing success ratio: {}".format(result['testing_accuracy']))
print("preduction of datapoints:")
print("ground truth: {}".format(map_label_to_class_name(datapoints[1], qsvm.label_to_class)))
print("prediction:   {}".format(result['predicted_classes']))
```

# Code Block 05 - Kernel Matrix
The output of this code block is a matrix of inner product values.  The matrix shows you visually the quality of the training process.  What you would like to see is clearly seprated regions of ligt and dark (our 2 lables), with a dark diagonal down the middle (these are the values where the inner product of the 2 feature vectors is 1).


```python
print("kernel matrix during the training:")
kernel_matrix = result['kernel_matrix_training']
img = plt.imshow(np.asmatrix(kernel_matrix),interpolation='nearest',origin='upper',cmap='bone_r')
plt.show()
```

# Code Block 06 - Breast Cancer Dataset - Toy Example

Run the algorithm with another dataset that has been provided for you. The goal is to distinguish breast cancer presence. The authors decided to use the first two principal components as features.  Principal Component Analysis (PCA) is a way to reduce the features in a complicated dataset.


```python
sample_Total, training_input, test_input, class_labels = Breast_cancer(training_size=20,
                                                                       test_size=10,
                                                                       n=2,
                                                                       PLOT_DATA=True)
```

# Code Block 07 - Run


```python
seed = 10598

feature_map = SecondOrderExpansion(num_qubits=feature_dim, depth=2, entanglement='linear')
qsvm = QSVMKernel(feature_map, training_input, test_input)

backend = Aer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024, seed=seed, seed_mapper=seed)

result = qsvm.run(quantum_instance)

"""declarative approach, re-use the params above
algo_input = SVMInput(training_input, test_input)
result = run_algorithm(params, algo_input)
"""
print("testing success ratio: ", result['testing_accuracy'])
```

# Code Block 08 - Kernel Matrix


```python
print("kernel matrix during the training:")
kernel_matrix = result['kernel_matrix_training']
img = plt.imshow(np.asmatrix(kernel_matrix),interpolation='nearest',origin='upper',cmap='bone_r')
plt.show()
```

# Installation Instructions for your local environment
Other tutorials can be downloaded by clicking [here](https://github.com/Qiskit/qiskit-tutorials/archive/master.zip) and to set them up follow the installation instructions [here](https://github.com/Qiskit/qiskit-tutorial/blob/master/INSTALL.md).
