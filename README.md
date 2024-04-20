# Documentation-Marvel-level-1
## TASK 1- Linear and Logistic Regression - HelloWorld for AIML
### Linear Regression- 
Linear regression analysis is used to predict the value of a variable based on the value of another variable. 
Linear regression predicts the relationship between two variables by assuming a linear connection between the independent and dependent variables.   
I split the california housing dataset into train and test sets (80% for training, and 20% for testing).       
Here is the code: https://github.com/vvvvvvss/Linear-regression/blob/main/linear%20(1).ipynb

### Logistic Regression-
Logistic regression estimates the probability of an event occuring. It provides a binary output, i.em 0 or 1.   
The data tells that there are 3 different species of iris     
setosa: represented by 0    
versicolor: represnted by 1     
virginica: represented by 2       
Predicting the spices of a given iris using the sepal width and length and petal width and length.
Each value we are predicting becomes the response here called the target.      
Here is the code: https://github.com/vvvvvvss/Logistic-regression/blob/main/Logistic%20regression.ipynb

## TASK 2 - Matplotlib and Data Visualisation
Matplotlib is a plotting and data visualizing library for python programming. Using matplotlib, here are some plots on various datasets:      
#### Simple sine and cosine wave     
![task2](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/12d762d2-fd29-41ce-ae3b-a9aeadd0c123)

#### Line plot        
![task2 2](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/a856a9e5-3cf0-4e15-8e8c-e0db7f42b572)       
#### Scatter plot         
![task2 3](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/70df57fc-804d-4d2d-85d4-3536744a3f4b)     
#### Bar plot     
![task2 4 1](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/f1cfece6-ef8b-4316-94c6-36b6147df368)
      
#### Stacked bar plot    
![task2 4 2](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/ab55adfc-1d87-4adb-ba77-50d0b73471dc)
    
#### Voilen plot        
![task2 4 3](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/f04d59ea-32a6-4260-93a9-fd3fdd639d15)
     
#### Marginal plot    
![task2 4 4](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/da8d3daf-bd15-4ce6-8683-c611d15399a3)
    
#### 3D plot    
![task2 4 5](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/ee2a61ba-3614-4af2-b530-7cf0b66d6e01)
      
#### Filled Counter plot    
![task2 4 6 2](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/986e8645-5d68-4524-8742-900c985976e7)
    
#### Unfilled Counter plot    
![task2 4 6](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/090a9f4c-a0ce-49d7-a503-21945a262521)
    
#### Heat map plot    
![task2 4 7](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/e2bf1551-db87-4539-9a3d-b9a357550821)
        
#### Histogram plot
![task2 5 1](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/9f492a4b-053d-48b9-9b11-c22b99e2391e)
       
Here is the code: https://github.com/vvvvvvss/Matplotlib-and-Data-Visualisation


## TASK 3 - Numpy
NumPy is a library for the Python programming language, that adds support for large and multi-dimensional arrays, along with a large collection of high-level mathematical functions to operate on these arrays. A feature of NumPy used here is the repeat function. Using the `np.repeat` function elements of the array can be repeated along different axises. NumPy can also be used to arrange the elements of the array in ascending order using the function `np.argsort`.  
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/930c060e-fb98-493f-868d-83970d9a1d26)
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/7013fbb6-68c2-4973-9dd5-7d7cc98c1080)


## Task 4 - Metrics and Performance Evaluation
To evaluate the performance or quality of the model, different metrics are used, and these metrics are known as performance metrics or evaluation metrics.      
### Regression Matrices
Regression matrices are supervised machine learning models. Some common regression models are   
MSE - Mean Squared Error:   
It measures the average squared difference between the actual and predicted values. Lower values indicate better performance.The squaring is critical to reduce the complexity with negative signs. To minimize MSE, the model could be more accurate, which would mean the model is closer to actual data.           
![Etuc3lBXcAEH7wO](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/355e3c60-7d28-4737-9bce-739e155c2896)       
MAE - Mean Absolute Error:     
It calculates the average absolute differences between the actual and predicted values.       
![1_m0O6zXxx9v8S3b1buUWBog](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/58133c62-b955-4b0a-95ce-75b965e58d8c)
R2 - R-Squared Error:
R-squared is a statistical measure that represents the goodness of fit of a regression model. The value of R-square lies between 0 to 1. Where we get R-square equals 1 when the model perfectly fits the data and there is no difference between the predicted value and actual value.        
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/53daa9c3-b179-4165-aca8-5fb91b63cb3b)         
Here is the code: https://github.com/vvvvvvss/Regression-matrices.
### Classification Matrices
In machine learning, classification is the process of categorizing a given set of data into different categories. For classification we make use of a confusion matrix. It is a mean of displaying the number of accurate and inaccurate instances based on the model’s predictions.      
he matrix displays the number of instances produced by the model on the test data.

True positives (TP): occur when the model accurately predicts a positive data point.       
True negatives (TN): occur when the model accurately predicts a negative data point.     
False positives (FP): occur when the model predicts a positive data point incorrectly.    
False negatives (FN): occur when the model predicts a negative data point incorrectly.    
![1_Z54JgbS4DUwWSknhDCvNTQ](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/dd334bba-a737-4545-9628-33e61b23bcb4)        

Here is the code: https://github.com/vvvvvvss/classification-matrices/blob/main/code.py

## Task 5 - Linear And Logistic Regression from scratch
Linear And Logistic Regression from scratch implies building an algorthm for the same without relaying on pre-built libraries or functions. This involves learning of the mathematical aspects behind the algorthm.    
### Linear regression 
aims at finding the best fit straight line that passes through the given data. This straight line provides the relationship between the independent variables and dependent variables. Simple linear regression can be written as      
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/d0c97d2c-d500-4ab1-b100-8a1f8b200bd1)     
The cost function, also known as the loss function, measures the difference between the predicted values and the actual values. In linear regression, the commonly used cost function is Mean Squared Error (MSE).     
Here is the code: https://github.com/vvvvvvss/Linear_regression_from_SCRATCH      
    
### Logistic regression 
models the probability that a given input belongs to a particular category. It's commonly used for binary classification problems, where the output variable has two possible outcomes, i.e, 0 or 1.  In logistic regression, the logistic function which is also a sigmoid function is used to map the input features to probabilities between 0 and 1. The logistic function is defined as:   
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/4b334730-f393-4058-affb-7e2d7637a423)     
Here is the code: https://github.com/vvvvvvss/Logistic_Regression_from_SCRATCH



## Task 6 - K-Nearest Neighbors
K-Nearest Neighbors (KNN) is a simple yet powerful algorithm used for both classification and regression tasks in machine learning. It's a type of instance-based learning, also known as lazy learning, where the algorithm doesn't explicitly build a model. Instead, it memorizes the entire training dataset and makes predictions based on the similarity of new instances to the existing data points. KNN calculates the distance between the new instance (or query point) and every point in the training dataset. Common distance metrics used include Euclidean distance. After calculating distances, KNN identifies the K nearest neighbors to the new instance. KNN often employs a simple majority voting rule for classification. In regression, the predicted value is the mean (or weighted mean) of the target values of the K nearest neighbors. To evaluate the performance of the KNN model, you typically use techniques such as cross-validation, where you partition the dataset into training and testing sets and measure metrics like accuracy, precision, recall, F1-score (for classification), or Mean Squared Error (MSE) (for regression).     
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/4522f6b7-837c-450c-bc4e-301abc0dac34)
       


### KNN from scratch    
Euclidean distance is used to calculate the distance between the new instance (or query point) and every point in the training dataset. The dataset is then loaded, if necessary. Ensure that the data is in a format suitable for distance calculations. Use majority voting to assign the class label to the new instance. Take the average (or weighted average) of the target values of the K nearest neighbors for regression.

Here is the code: https://github.com/vvvvvvss/K--Nearest-Neighbor-Algorithm

## Task 7 - An elementary step towards understanding Neural Network
Neural networks are computational systems inspired by the structure and functioning of the human brain. They are the fundamental component of many Machine learning models. One can find their applicaion in image detection, speech recognition etc. ANNs and CNNs come under the types of neural networks.       
Here is the blog: https://github.com/vvvvvvss/Neural-networks/blob/main/NeuralNetworks.md      
Large Language Models aka LLMs are built on machine learning, specifically a type of neural network called a transformer model. They can recognize and generate text, and can be used for a number of tasks including writing code, summarizing, translation.        
Here is the blog: https://github.com/vvvvvvss/Neural-networks/blob/main/LLMs.md


## Task 8 - Mathematics behind machine learning
Curve fitting: Curve fitting is the process of constructing a curve, or mathematical function, that has the best fit to a series of data points.        
Here is the code: https://github.com/vvvvvvss/curve-fitting             
Fourier Transforms: Fourier Transform is a mathematical model which helps to transform the signals between two different domains, such as transforming signal from frequency domain to time domain or vice versa. 
Fourier transforms are found in almost everything these days, from digital music to quantum mechanics to image recognition. In simple terms, a fourier transfomer simplifies a wave into a sum of sine and cosine waves         
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/895dc818-2c1e-4afc-a4f2-4e250fe8491a)            

Here is the code: [https://github.com/vvvvvvss/Fourier-transformers](https://github.com/vvvvvvss/Fourier-transformers/blob/main/Fourier.ipynb)


# Task 9: Data Visualization for Exploratory Data Analysis
Plotly is a dynamic tool, much better than others like matplotlib and seaborn for data visualization. Here is a scatter, bar, histogram and a box plot on the classic Iris dataset and a 3D plot on a random data: https://github.com/vvvvvvss/data-visualization/blob/main/datavisualization.ipynb

# Task 10: An introduction to Decision Trees
Decision trees are a branch of superivised machine learning. It is an important classification and regression tool. Here is a prediction of salary analyzing parameters like the company, company location, work year, job title etc: 
https://github.com/vvvvvvss/decisiontrees/blob/main/decisiontrees.ipynb

# Task 11: Exploration of a Real world application of Machine Learning
Machine learning (ML) can be used to predict traffic flow and congestion patterns in real-time based on large amounts of traffic data. These predictions can help improve transportation systems and reduce economic losses. 
Here is the case study: https://github.com/vvvvvvss/case-study
