
# Example 01 - Quantum Support Vector Machines (qSVM)


***
#### Thank you to these original authors:
Vojtech Havlicek<sup>[1]</sup>, Kristan Temme<sup>[1]</sup>, Antonio Córcoles<sup>[1]</sup>, Peng Liu<sup>[1]</sup>, Richard Chen<sup>[1]</sup>, Marco Pistoia<sup>[1]</sup> and Jay Gambetta<sup>[1]</sup>

#### References and additional details:

Vojtech Havlicek, Antonio D. C´orcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, and Jay M. Gambetta1, "Supervised learning with quantum enhanced feature spaces," [arXiv: 1804.11326](https://arxiv.org/pdf/1804.11326.pdf)


#### Quantum Computing Framework
- <sup>[1]</sup>IBM QISKit


# Running the Code

***

Installing everything you need on your own laptop takes a little time, so for right now the easiest way to get started is to use a "Binder image".  This setup lets you use the code notebooks via the web IMMEDIATELY. I think this is your best approach, for now, because it means you can run the code today to get a feel for how this example of a Quantum Machine Learning algorithm works. In the future, you can follow the full installation process (2 hours) and run your own code. Let's get started. 

# Overview

1) Click on [the Binder image](https://mybinder.org/v2/gh/qiskit/qiskit-tutorials/master?filepath=index.ipynb).  (2 min)

2) You'll need an account and API token from https://quantumexperience.ng.bluemix.net/qx (5 min)

3) Scroll down to 1.6 and click on *Artificial Intelligence*.

4) On this next page, select *Quantum SVM for Classification*.

3) Run the Jupyter Notebook. If this is your first time using a Jupyter Notebook, select a cell (it will now have a blue border).  Now SHIFT then RETURN to run the code on the remote server.  The notes below will add explanations and a lot more context to the published notebooks. I thought this was a much more efficient way to help you because you can always link to the newest published code, download it, and use these notes to modify as you wish.  



# Code Block 01


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

# Code Block 02

### [Optional] Setup token to run the experiment on a real device
If you would like to run the experiement on a real device, you need to setup your account first.

Note: If you do not store your token yet, use `IBMQ.save_accounts()` to store it first.


```python
from qiskit import IBMQ
IBMQ.load_accounts()
```

First we prepare the dataset, which is used for training, testing and the finally prediction.

*Note: You can easily switch to a different dataset, such as the Breast Cancer dataset, by replacing 'ad_hoc_data' to 'Breast_cancer' below.*

# Code Block 03


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

With the dataset ready we initialize the necessary inputs for the algorithm:
- the input dictionary (params) 
- the input object containing the dataset info (algo_input).

With everything setup, we can now run the algorithm.

For the testing, the result includes the details and the success ratio.

For the prediction, the result includes the predicted labels. 

# Code Block 04


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

# Code Block 05


```python
print("kernel matrix during the training:")
kernel_matrix = result['kernel_matrix_training']
img = plt.imshow(np.asmatrix(kernel_matrix),interpolation='nearest',origin='upper',cmap='bone_r')
plt.show()
```

# Code Block 06

### The breast cancer dataset
Now we run our algorithm with the real-world dataset: the breast cancer dataset, we use the first two principal components as features.


```python
sample_Total, training_input, test_input, class_labels = Breast_cancer(training_size=20,
                                                                       test_size=10,
                                                                       n=2,
                                                                       PLOT_DATA=True)
```

# Code Block 07


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

# Code Block 08


```python
print("kernel matrix during the training:")
kernel_matrix = result['kernel_matrix_training']
img = plt.imshow(np.asmatrix(kernel_matrix),interpolation='nearest',origin='upper',cmap='bone_r')
plt.show()
```

# Installation Instructions for your local environment
Other tutorials can be downloaded by clicking [here](https://github.com/Qiskit/qiskit-tutorials/archive/master.zip) and to set them up follow the installation instructions [here](https://github.com/Qiskit/qiskit-tutorial/blob/master/INSTALL.md).
