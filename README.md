# ML-House-Prices-Advanced-Regression-Techniques
Predict sales prices and practice feature engineering, RFs, and gradient boosting

## XGB
XGBoost (eXtreme Gradient Boosting) is a machine learning algorithm that is used for supervised learning tasks such as regression and classification. It is an implementation of the gradient boosting decision tree algorithm.

XGBoost is a popular algorithm due to its high performance and scalability, making it suitable for large-scale datasets. It works by building an ensemble of decision trees, where each tree is trained on the residual errors of the previous tree.

The algorithm uses a gradient descent approach to optimize the objective function, which is the sum of the loss function and a regularization term. The loss function is typically a measure of the difference between the predicted and actual values, and the regularization term is used to prevent overfitting.

XGBoost has several advanced features that make it stand out from other gradient boosting algorithms. For example, it uses a distributed computing framework to parallelize the training process, which can significantly reduce the training time. It also includes built-in handling of missing values, automatic feature selection, and early stopping to prevent overfitting.

XGBoost has achieved state-of-the-art results in many machine learning competitions and is widely used in industry for a variety of applications such as credit risk modeling, fraud detection, and image classification. It has also been successfully applied to non-tabular data such as text and images by adapting the algorithm to work with different types of input data.
## Decision Tree
A Decision Tree is a type of supervised learning algorithm used in machine learning and data mining for classification and regression analysis. It is a tree-like model that is constructed by recursively splitting the data into subsets based on the most important features and their values.

The decision tree algorithm begins by selecting a feature from the dataset that best separates the data into two or more homogeneous subsets. The selected feature and its value become the root node of the tree. The algorithm then recursively applies the same process to each subset until a stopping criterion is met, such as a maximum tree depth or a minimum number of samples per leaf.

The decision tree model is easy to interpret and can handle both categorical and numerical data. It can also handle missing values and outliers.

In classification tasks, the decision tree algorithm is used to create a tree of rules that can be used to predict the class of a new sample. In regression tasks, the algorithm creates a tree that predicts the target value for new samples.

Decision trees are widely used in many applications, including finance, healthcare, and marketing. They can be used to predict customer churn, detect credit card fraud, and diagnose medical conditions, among others. However, decision trees are prone to overfitting, especially when the tree becomes too complex. To avoid overfitting, techniques such as pruning and ensemble methods, such as Random Forest, can be used.
## ANN
ANN stands for Artificial Neural Network, which is a type of machine learning algorithm inspired by the structure and function of biological neurons in the human brain. ANNs are used for supervised learning tasks such as classification and regression.

In an ANN, the input data is passed through a network of interconnected nodes, which are organized into layers. Each node in the network is connected to other nodes in the adjacent layers and each connection is assigned a weight. The weights are adjusted during the training process to optimize the network's performance.

The first layer of an ANN is the input layer, which receives the input data. The intermediate layers are called hidden layers, which perform transformations on the input data to extract relevant features. The final layer is the output layer, which produces the predictions or classifications.

ANNs can have different architectures such as feedforward, recurrent, and convolutional. In a feedforward network, the data flows only in one direction from the input layer to the output layer, while in a recurrent network, the data can flow in loops, allowing the network to handle time-series data. Convolutional networks are used primarily for image and video data and have convolutional layers that apply filters to the input data to detect features.

ANNs have several advantages such as the ability to learn complex non-linear relationships between inputs and outputs, and the ability to generalize well to unseen data. They have been applied to a variety of applications such as image and speech recognition, natural language processing, and robotics. However, ANNs can also be computationally expensive to train and require large amounts of data to achieve good performance.
