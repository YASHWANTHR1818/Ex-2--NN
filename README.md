<H3>Name: YASHWANTH R </H3>
<H3>Register no. 212224240188 </H3>

<H3>Experiment No. 2 </H3>

## Implementation of Perceptron for Binary Classification

# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:

A Perceptron is a basic machine learning algorithm introduced by Frank Rosenblatt in 1959. It is inspired by the working of a biological neuron.

It is a supervised learning algorithm used for binary classification. It assigns weights to input features and predicts the class of a sample.

Working
Training: Initialize weights, predict the class, compare it with the actual class, and update the weights if the prediction is wrong.
Testing: Use the trained model to predict the class of new samples.
Important Points
Use sufficient labeled data.
Preprocess and scale the data when required.
Perceptron works well with linearly separable data.
Mathematical Model
f(x)=w⋅x+b

Where x is the input, w is the weight, and b is the bias.

The predicted class is:

y
^
	​

={
1,
0,
	​

w⋅x+b≥0
w⋅x+b<0
	​


A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

 


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>

# PROGRAM:
```PYTHON
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
```PYTHON
class Perceptron:
    def __init__(self, learning_rate = 0.1):
        self.learning_rate = learning_rate
        self.b = 0.0
        self.w = None
        self.misclassified_samples = []

    def fit(self, x: np.array, y: np.array, n_iter = 10):
        self._b = 0.0
        self._w = np.zeros(x.shape[1])
        self.misclassified_samples = []
        for _ in range(n_iter):
            errors = 0
            for xi, yi in zip(x, y):
                update = self.learning_rate * (yi - self.predict(xi))
                self._b += update
                self._w += update * xi
                errors += int(update != 0.0)
            self.misclassified_samples.append(errors)

    def f(self, x: np.array) -> float:
        return np.dot(x, self._w) + self._b

    def predict(self, x: np.array) -> int:
        return np.where(self.f(x) >= 0, 1, -1)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
print(df.head())
y = df.iloc[:, 4].values
x = df.iloc[:, 0:3].values
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Iris data set')
ax.set_xlabel("Sepal length in width (cm)")
ax.set_ylabel("Sepal width in width (cm)")
ax.set_zlabel("Petal length in width (cm)")
ax.scatter(x[:50,0], x[:50,1], x[:50,2], color='red',marker='o', s=4, label="Iris Setosa")
ax.scatter(x[50:100,0], x[50:100,1], x[50:100,2], color='blue',marker='^', s=4, label="Iris Versicolour")
ax.scatter(x[100:150,0], x[100:150,1], x[100:150,2], color='green',marker='x', s=4, label="Iris Virginica")
plt.legend(loc='upper left')
plt.show()
x = x[0:100, 0:2] 
y = y[0:100]
plt.figure(figsize=(4,4))
plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x',label='Versicolour')
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.legend(loc='upper left')
plt.show()
y = np.where(y == 'Iris-setosa', 1, -1)
x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=0)
classifier = Perceptron(learning_rate=0.01)
classifier.fit(x_train, y_train)
print("accuracy", accuracy_score(classifier.predict(x_test), y_test)*100)
plt.figure(figsize=(4,4))
plt.plot(range(1, len(classifier.misclassified_samples) + 1),classifier.misclassified_samples, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.show()
```

# OUTPUT:

<img width="520" height="600" alt="image" src="https://github.com/user-attachments/assets/7130a04d-c8a4-4ce2-b673-301464495a9c" />
  
<img width="521" height="432" alt="image" src="https://github.com/user-attachments/assets/2a0ad7d7-0c68-46db-95f0-47f6c24579e7" />

<img width="517" height="429" alt="image" src="https://github.com/user-attachments/assets/f3973d0f-86f4-4709-bea2-926b601dfb7a" />

# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
