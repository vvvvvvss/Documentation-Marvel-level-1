# Documentation-Marvel-level-1
## TASK 1- Linear and Logistic Regression - HelloWorld for AIML
### Linear Regression- Linear regression analysis is used to predict the value of a variable based on the value of another variable. 
Linear regression predicts the relationship between two variables by assuming a linear connection between the independent and dependent variables.   
I split our dataset into train and test sets (80% for training, and 20% for testing).

### Logistic Regression-
The data tells that there are 3 different species of iris
setosa: represented by 0
versicolor: represnted by 1
virginica: represented by 2
Predicting the spices of a given iris using the sepal width and length and petal width and length.
Each value we are predicting becomes the response here called the target.

## TASK 2 Matplotlib and Data Visualisation
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/blob/main/task2.png)
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/blob/main/task2.2.png)
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/blob/main/task2.3.png)
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/blob/main/task2.4.1.png)
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/blob/main/task2.4.2.png)
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/blob/main/task2.4.3.png)
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/blob/main/task2.4.4.png)
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/blob/main/task2.4.5.png)
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/blob/main/task2.4.6.2.png)      
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/blob/main/task2.4.6.png)
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/blob/main/task2.4.7.png)
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/blob/main/task2.5.1.png)



## TASK 3 - Numpy
NumPy is a library for the Python programming language, that adds support for large and multi-dimensional arrays, along with a large collection of high-level mathematical functions to operate on these arrays. A feature of NumPy used here is the repeat function. Using the `np.repeat` function elements of the array can be repeated along different axises. NumPy can also be used to arrange the elements of the array in ascending order using the function `np.argsort`.  
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/930c060e-fb98-493f-868d-83970d9a1d26)
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/7013fbb6-68c2-4973-9dd5-7d7cc98c1080)


## Task 4 - Metrics and Performance Evaluation
Metrices helps in evaluating performenaces. Regression matrices are supervised machine learning models. Some common regression models are   
MSE - Mean Squared Error:   
It measures the average squared difference between the actual and predicted values. Lower values indicate better performance.The squaring is critical to reduce the complexity with negative signs. To minimize MSE, the model could be more accurate, which would mean the model is closer to actual data.           
![Etuc3lBXcAEH7wO](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/355e3c60-7d28-4737-9bce-739e155c2896)       
MAE - Mean Absolute Error:     
It calculates the average absolute differences between the actual and predicted values.       
![1_m0O6zXxx9v8S3b1buUWBog](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/58133c62-b955-4b0a-95ce-75b965e58d8c)
R2 - R-Squared Error:
This metric explains the proportion of the variance in the dependent variable that is predictable from the independent variables. R-squared values range from 0 to 1, where 1 indicates a perfect fit.
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/53daa9c3-b179-4165-aca8-5fb91b63cb3b)         
Here is the code: https://github.com/vvvvvvss/Regression-matrices

## Task 5 - Linear And Logistic Regression from scratch
Linear And Logistic Regression from scratch implies building an algorthm for the same without relayimg on pre-built libraries or functions. This involves learning of the mathematical aspects behind the algorthm.    
Linear regression aims at finding the best fit straight line that passes through the given data. This straight line provides the relationship between the independent variables and dependent variables. Simple linear regression can be written as      
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/d0c97d2c-d500-4ab1-b100-8a1f8b200bd1)     
The cost function, also known as the loss function, measures the difference between the predicted values and the actual values. In linear regression, the commonly used cost function is Mean Squared Error (MSE).     
Here is the code: https://github.com/vvvvvvss/Linear_regression_from_SCRATCH      
Logistic regression models the probability that a given input belongs to a particular category. It's commonly used for binary classification problems, where the output variable has two possible outcomes, i.e, 0 or 1.  In logistic regression, the logistic function which is also a sigmoid function is used to map the input features to probabilities between 0 and 1. The logistic function is defined as:   
![image](https://github.com/vvvvvvss/Documentation-Marvel-level-1/assets/148562671/4b334730-f393-4058-affb-7e2d7637a423)     
Here is the code: https://github.com/vvvvvvss/Logistic_Regression_from_SCRATCH



## Task 6 - K-Nearest Neighbors
K-Nearest Neighbors (KNN) is a simple yet powerful algorithm used for both classification and regression tasks in machine learning. It's a type of instance-based learning, also known as lazy learning, where the algorithm doesn't explicitly build a model. Instead, it memorizes the entire training dataset and makes predictions based on the similarity of new instances to the existing data points. KNN calculates the distance between the new instance (or query point) and every point in the training dataset. Common distance metrics used include Euclidean distance. After calculating distances, KNN identifies the K nearest neighbors to the new instance. KNN often employs a simple majority voting rule for classification. In regression, the predicted value is the mean (or weighted mean) of the target values of the K nearest neighbors. To evaluate the performance of the KNN model, you typically use techniques such as cross-validation, where you partition the dataset into training and testing sets and measure metrics like accuracy, precision, recall, F1-score (for classification), or Mean Squared Error (MSE) (for regression).     
### KNN from scratch    
Euclidean distance is used to calculate the distance between the new instance (or query point) and every point in the training dataset. The dataset is then loaded, if necessary. Ensure that the data is in a format suitable for distance calculations. Use majority voting to assign the class label to the new instance. Take the average (or weighted average) of the target values of the K nearest neighbors for regression.     
Here is the code: https://github.com/vvvvvvss/K--Nearest-Neighbor-Algorithm


​


