# Stock Prediction Using Time-Series-Forecasting
\
Hello! Welcome to the WiDS2022 Project ***"Stock Market Prediction using Time Series Forecasting"***

## Project Introduction
Our journey of stock market analysis will begin all the way from learning the theoretical basics of python and machine learning to the hands on application of the knowledge to generate useful price predictions. Stock Price is a time stamped data which can be subject to autocorrelation, seasonality and stationarity. Time Series data can be modeled using techniques like moving average, exponential smoothing, ARIMA, etc. In the duration of the project, we shall better understand the jargon used above and then implement the ARIMA model to predict and test our model. 

Googling is the number one friend of any programmer. Although we shall have a WhatsApp Group for doubts and discussions yet I'll encourage you to have atleast 20 tabs worth of search before asking for help (PS - Everyone has a single lifetime so don't waste ages ofc!)

## Tentative Schedule 

Week 1 - Setting up Jupyter Notebook and learn the basics of Python Language.

Week 2 - Learning the theoretical basis of Machine Learning in general and Time Series Data in particular.

Week 3 - Getting familiar with common AI/ML libraries and begin model implementation.

Week 4 - Complete the model implementation and learn about good coding practices

Week 5 - Debugging and Finalization of project along with discussions on further learnings.

## Week 1

[Install Anaconda](https://docs.anaconda.com/anaconda/install/) - Anaconda is one of the most beloved tool for data scientist because it comes packed with a collection of over 7,500+ open-source python/R packages.

Data Science/Analytics can be performed with multiple languages like R/Matlab/Python. We shall stick to python in this project since it's a beginner friendly language and is widely used globally. Here are some good resources to learn python for data science purpose:

(PS - Understanding different data types like **lists**, **tuples**, **dictionaries** and **sets** helps in appreciating the benefit libraries)

[W3School's Python Articles](https://www.w3schools.com/python/) - Best for students with prior programming experience.

[Python Tutorial by Mosh Hamadani](https://www.youtube.com/watch?v=kqtD5dpn9C8&ab_channel=ProgrammingwithMosh) - More beginner friendly tutorial

*A basic understanding of datatypes, logical expressions, loops, lists and functions is enough to begin with*

## Week 2

And Finally, we'll start with a basic introduction of Machine Learning
 
Machine Learning is the field of study that gives computers the capability to learn without being explicitly programmed. ML is one of the most exciting technologies that one would have ever come across. As it is evident from the name, it gives the computer that makes it more similar to humans: The ability to learn. Machine learning is actively being used today, perhaps in many more places than one would expect.
From translation apps to autonomous vehicles, all powers with Machine Learning. It offers a way to solve problems and answer complex questions. It is basically a process of training a piece of software called an algorithm or model, to make useful predictions from data

#### Types of machine learning problems

###### Supervised learning:
The model or algorithm is presented with example inputs and their desired outputs and then finding patterns and connections between the input and the output. The goal is to learn a general rule that maps inputs to outputs. The training process continues until the model achieves the desired level of accuracy on the training data.Some real-life examples are:
Image Classification: You train with images/labels. Then in the future you give a new image expecting that the computer will recognize the new object.
Market Prediction/Regression: You train the computer with historical market data and ask the computer to predict the new price in the future. 

###### Unsupervised learning:
Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets consisting of input data without labeled responses. In unsupervised learning algorithms, classification or categorization is not included in the observations.

#### Different Machine Learning Models

#### Classification
As the name suggests, Classification is the task of “classifying things” into sub-categories. But, by a machine! If that doesn’t sound like much, imagine your computer being able to differentiate between you and a stranger. Between a potato and a tomato. Between an A grade and an F. Now, it sounds interesting now. In Machine Learning and Statistics, Classification is the problem of identifying to which of a set of categories (subpopulations), a new observation belongs, on the basis of a training set of data containing observations and whose categories membership is known.

Types of classification :
Binary Classification: When we have to categorize given data into 2 distinct classes. Example – On the basis of given health conditions of a person, we have to determine whether the person has a certain disease or not.
Multiclass Classification: The number of classes is more than 2. For Example – On the basis of data about different species of flowers, we have to determine which specie our observation belongs.

#### Regression
A regression problem is when the output variable is a real or continuous value, such as “salary” or “weight”. Many different models can be used, the simplest is linear regression. It tries to fit data with the best hyperplane which goes through the points.
Regression Analysis is a statistical process for estimating the relationships between the dependent variables or criterion variables and one or more independent variables or predictors. Regression analysis explains the changes in criteria in relation to changes in select predictors. The conditional expectation of the criteria is based on predictors where the average value of the dependent variables is given when the independent variables are changed. Three major uses for regression analysis are determining the strength of predictors, forecasting an effect, and trend forecasting. 
##### Linear Regression
Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting. Different regression models differ based on – the kind of relationship between dependent and independent variables they are considering, and the number of independent variables getting used. There are many names for a regression’s dependent variable.  It may be called an outcome variable, criterion variable, endogenous variable, or regressand.  The independent variables can be called exogenous variables, predictor variables, or regressors.Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x). So, this regression technique finds out a linear relationship between x (input) and y(output). Hence, the name is Linear Regression. In the figure above, X (input) is the work experience and Y (output) is the salary of a person. The regression line is the best fit line for our model.While training the model we are given : x: input training data (univariate – one input variable(parameter)) y: labels to data (Supervised learning) When training the model – it fits the best line to predict the value of y for a given value of x. The model gets the best regression fit line by finding the best θ1 and θ2 values. θ1: intercept θ2: coefficient of x Once we find the best θ1 and θ2 values, we get the best fit line. So when we are finally using our model for prediction, it will predict the value of y for the input value of x.
Cost Function (J): By achieving the best-fit regression line, the model aims to predict y value such that the error difference between predicted value and true value is minimum. So, it is very important to update the θ1 and θ2 values, to reach the best value that minimize the error between predicted y value (pred) and true y value (y).
Cost function(J) of Linear Regression is the Root Mean Squared Error (RMSE) between predicted y value (pred) and true y value (y). Gradient Descent: To update θ1 and θ2 values in order to reduce Cost function (minimizing RMSE value) and achieving the best fit line the model uses Gradient Descent. The idea is to start with random θ1 and θ2 values and then iteratively updating the values, reaching minimum cost.
Go through the following [video](https://youtu.be/nk2CQITm_eo) for better understanding.

##### Gradient Descent in Linear Regression
In linear regression, the model targets to get the best-fit regression line to predict the value of y based on the given input value (x). While training the model, the model calculates the cost function which measures the Root Mean Squared error between the predicted value (pred) and true value (y). The model targets to minimize the cost function. 
To minimize the cost function, the model needs to have the best value of θ1 and θ2. Initially model selects θ1 and θ2 values randomly and then iteratively update these value in order to minimize the cost function until it reaches the minimum. By the time model achieves the minimum cost function, it will have the best θ1 and θ2 values. Using these finally updated values of θ1 and θ2 in the hypothesis equation of linear equation, the model predicts the value of x in the best manner it can. 
Therefore, the question arises – How do θ1 and θ2 values get updated? Go through the following [link](https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/) to understand how gradient descent works and its implementation.
Go through the following [video](https://youtu.be/sDv4f4s2SB8) for better understanding.

##### Logistic Regression
Logistic regression is basically a supervised classification algorithm. In a classification problem, the target variable(or output), y, can take only discrete values for a given set of features(or inputs), X.
Contrary to popular belief, logistic regression is a regression model. The model builds a regression model to predict the probability that a given data entry belongs to the category numbered as “1”. Just like Linear regression assumes that the data follows a linear function, Logistic regression models the data using the [sigmoid function](https://images.app.goo.gl/LiHaGFmatgmKp3WGA). Logistic regression becomes a classification technique only when a decision threshold is brought into the picture. The setting of the threshold value is a very important aspect of Logistic regression and is dependent on the classification problem itself. The decision for the value of the threshold value is majorly affected by the values of precision and recall. Ideally, we want both precision and recall to be 1, but this seldom is the case.
In the case of a Precision-Recall tradeoff, we use the following arguments to decide upon the threshold:-
1. Low Precision/High Recall: In applications where we want to reduce the number of false negatives without necessarily reducing the number of false positives, we choose a decision value that has a low value of Precision or a high value of Recall. For example, in a cancer diagnosis application, we do not want any affected patient to be classified as not affected without giving much heed to if the patient is being wrongfully diagnosed with cancer. This is because the absence of cancer can be detected by further medical diseases but the presence of the disease cannot be detected in an already rejected candidate.
2. High Precision/Low Recall: In applications where we want to reduce the number of false positives without necessarily reducing the number of false negatives, we choose a decision value that has a high value of Precision or a low value of Recall. For example, if we are classifying customers whether they will react positively or negatively to a personalized advertisement, we want to be absolutely sure that the customer will react positively to the advertisement because otherwise, a negative reaction can cause a loss of potential sales from the customer.
Based on the number of categories, Logistic regression can be classified as: 
binomial: target variable can have only 2 possible types: “0” or “1” which may represent “win” vs “loss”, “pass” vs “fail”, “dead” vs “alive”, etc.
multinomial: target variable can have 3 or more possible types which are not ordered(i.e. types have no quantitative significance) like “disease A” vs “disease B” vs “disease C”.
ordinal: it deals with target variables with ordered categories. For example, a test score can be categorized as:“very poor”, “poor”, “good”, “very good”. Here, each category can be given a score like 0, 1, 2, 3.
Go through the following [playlist](https://youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe) for better understanding.

##### Decision Tree
Decision Tree is the most powerful and popular tool for classification and prediction. A Decision tree is a flowchart-like tree structure, where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label.
Construction of Decision Tree: A tree can be “learned” by splitting the source set into subsets based on an attribute value test. This process is repeated on each derived subset in a recursive manner called recursive partitioning. The recursion is completed when the subset at a node all has the same value of the target variable, or when splitting no longer adds value to the predictions. The construction of a decision tree classifier does not require any domain knowledge or parameter setting, and therefore is appropriate for exploratory knowledge discovery. Decision trees can handle high-dimensional data. In general decision tree classifier has good accuracy. Decision tree induction is a typical inductive approach to learn knowledge on classification.Decision trees classify instances by sorting them down the tree from the root to some leaf node, which provides the classification of the instance. An instance is classified by starting at the root node of the tree, testing the attribute specified by this node, then moving down the tree branch corresponding to the value of the attribute. This process is then repeated for the subtree rooted at the new node.
Gini Index: Gini Index is a score that evaluates how accurate a split is among the classified groups. Gini index evaluates a score in the range between 0 and 1, where 0 is when all observations belong to one class, and 1 is a random distribution of the elements within classes. In this case, we want to have a Gini index score as low as possible. Gini Index is the evaluation metrics we shall use to evaluate our Decision Tree Model.
Go through the following [video](https://youtu.be/_L39rN6gz7Y) for better understanding

##### Random Forest
Every decision tree has high variance, but when we combine all of them together in parallel then the resultant variance is low as each decision tree gets perfectly trained on that particular sample data, and hence the output doesn’t depend on one decision tree but on multiple decision trees. In the case of a classification problem, the final output is taken by using the majority voting classifier. In the case of a regression problem, the final output is the mean of all the outputs. This part is called Aggregation. Random Forest is an ensemble technique capable of performing both regression and classification tasks with the use of multiple decision trees and a technique called Bootstrap and Aggregation, commonly known as bagging. The basic idea behind this is to combine multiple decision trees in determining the final output rather than relying on individual decision trees. 
Random Forest has multiple decision trees as base learning models. We randomly perform row sampling and feature sampling from the dataset forming sample datasets for every model. This part is called Bootstrap.
Go through the following [playlist](https://youtube.com/playlist?list=PLblh5JKOoLUIE96dI3U7oxHaCAbZgfhHk) for better understanding.

##### Support Vector Machines
In machine learning, support vector machines (SVMs, also support vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples.An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. In addition to performing linear classification, SVMs can efficiently perform a non-linear classification, implicitly mapping their inputs into high-dimensional feature spaces.Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier. 
Go through the following [video](https://youtu.be/8A7L0GsBiLQ) for better understanding

### Time Series Forecasting
In Machine Learning, Time Series Forecasting refers to the use of statistical models to predict future values using the previously recorded observations.The underlying intention of time series forecasting is determining how target variables will change in the future by observing historical data from the time perspective, defining the patterns, and yielding short or long-term predictions on how change occurs – considering the captured patterns.
Go through the following [article](https://www.tableau.com/learn/articles/time-series-forecasting#:~:text=Time%20series%20forecasting%20occurs%20when,drive%20future%20strategic%20decision%2Dmaking.) to understand time series forecasting.

#### Different models of Time series forecasting
Time series models are used to forecast events based on verified historical data. Common types include ARIMA, smooth-based, and moving average. Not all models will yield the same results for the same dataset, so it's critical to determine which one works best based on the individual time series.

Modern market pace calls for a respective competitive edge. That is, if you are looking to take a sturdy, profitable position amongst business competitors in your niche. For this, you can (and should) use proper technologies that can help you stay at least one step ahead of your competition. Data forecasting has come a long way since formidable data processing-boosting technologies such as machine learning were introduced. ML-based predictive models nowadays may consider time-dependent components — seasonality, trends, cycles, irregular components, etc. — to maximize the preciseness of data-driven predictions and forecasts. This is called machine learning forecasting and it can be most beneficially applied across all sorts of business aspects, including sales and demand forecasting, recruiting forecasting, weather forecasting, content consumption forecasting; predictive planning and maintenance, and more. 

Time Series is a certain sequence of data observations that a system collects within specific periods of time — e.g., daily, monthly, or yearly. The specialized models are used to analyze the collected time-series data — describe and interpret them, as well as make certain assumptions based on shifts and odds in the collection. These shifts and odds may include the switch of trends, seasonal spikes in demand, certain repetitive changes or non-systematic shifts in usual patterns, etc. All the previously, recently, and currently collected data is used as input for time series forecasting where future trends, seasonal changes, irregularities, and such are elaborated based on complex math-driven algorithms. And with machine learning, time series forecasting becomes faster, more precise, and more efficient in the long run. ML has proven to help better process both structured and unstructured data flows, swiftly capturing accurate patterns within massifs of data. It is safe to say that time series machine learning principles basically outperform the classical time series forecasting approach. Thus, traditional methods are limited to processing only previously collected, readily available demand history. In turn, ML autonomously defines points of interest in the unlimited flow of data to then align them with customer data insights at hand and conduct what-if analysis. This results in particularly efficient takes on stimulating the demand in the commercial sector, for instance. However, this intricate predictive approach is beneficially used across numerous aspects of business management and optimization in most various niches.

Pretty much any company or organization dealing with constantly generated data and the need to adapt to operational shifts and changes can use time series forecasting. Machine learning serves as the ultimate booster here, allowing to better handle:

Stock prices forecasting — the data on the history of stock prices combined with the data on both regular and irregular stock market spikes and drops can be used to gain insightful predictions of the most probable upcoming stock price shifts.

Demand and sales forecasting — customer behavior patterns data along with inputs from the history of purchases, timeline of demand, seasonal impact, etc., enable ML models to point out the most potentially demanded products and hit the spot in the dynamic market.

Web traffic forecasting — common data on usual traffic rates among competitor websites is bunched up with input data on traffic-related patterns in order to predict web traffic rates during certain periods.

Climate and weather prediction — time-based data is regularly gathered from numerous interconnected weather stations worldwide, while ML techniques allow to thoroughly analyze and interpret it for future forecasts based on statistical dynamics.

Demographic and economic forecasting — there are tons of statistical inputs in demographics and economics, which is most efficiently used for ML-based time-series predictions. As a result, the most fitting target audience can be picked and the most efficient ways to interact with that particular TA can be elaborated.

Scientific studies forecasting — ML and deep learning principles accelerate the rates of polishing up and introducing scientific innovations dramatically. For instance, science data that requires an indefinite number of analytical iterations can be processed much faster with the help of patterns automated by machine learning.

How exactly does machine learning forecasting time series work in practice? Plenty of methods have been introduced over the years. Some of them may even be deemed outdated by now. The so-called legacy time series forecasting methods either take more time and effort to implement or bring comparatively insufficient results (or both) as opposed to more recently introduced alternatives. However, they still may fit specific goals better than other approaches.
Then, there are classical methods which are well-tried-and-tested approaches that remain the default for most time series forecasting instances and are most widely used. The great thing is that they can be efficiently reinforced with the powers of ML to achieve much better results. Topical methods, on the other hand, are methods focused on particular situations and goals that fit specific forecasting scenarios.
As in any case of working with machine learning, ML-forecasting can be supervised (requiring specific data input to work) or unsupervised (ongoing, self-learning data processing mechanisms). The methods listed below can be of both natures interchangeably. 

##### Recurrent Neural Network (RNN)
RNNs process a time series step-by-step, maintaining an internal state from time-step to time-step. Neural networks are great in this application as they can learn the temporal dependence from the given data. And considering input sequences from the temporal perspective opens horizons for more precise predictions. However, the method is considered a legacy because the “education” of neural networks can be too time-consuming.

##### Long Short-Term Memory (LSTM)
It’s kind of RNN, but while maintaining RNN’s ability to learn the temporal dynamics of sequential data, LSTM can furthermore handle the vanishing and exploding gradients problem. Thus, complex multivariate data sequences can be accurately modeled, and the need to establish pre-specified time windows (which solves many tasks that feed-forward networks cannot solve). The downside of overly time-consuming supervised learning, however, remains.

##### Multi-Layer Perceptron (MLP)
Univariate models can be used to model univariate time series forecasting problems. Multivariate MLP models use multivariate data where there is more than one observation for each time step. Then, there are multistep MLP models — there is little difference to the MLP model in predicting a vector output that represents different output variables or a vector output that represents multiple time steps of one variable. This is a very widely used method that even outperforms LSTM in certain autoregression instances.

##### ARIMA
Autoregression employs observations collected during previous time steps as input data for making regression equations that help predict the value to be generated at the next time step. ARIMA or an AutoRegressive Integrated Moving Average model combines autoregression and moving average principles, making forecasts correspond to linear combinations of past variable values and forecast errors, being one of the most popular approaches due to that.

##### Bayesian Neural Network (BNN)
BNN models involve constructing a prior distribution and updating this distribution by conditioning on the actual data. This is particularly useful for financial data because of its volatile nature, as nonlinear time series forecasting with machine learning is enabled. BNN treats network weights or parameters as random variables, being among the most universally used models out there.

##### Convolutional Neural Network (CNN)
Although analysis of image datasets is considered their main field of application, convolutional neural networks can show even better results than RNNs in time series prediction cases involving other types of spatial data. For one thing, they learn faster, boosting the overall data processing performance. However, CNN’s can also be joined with RNNs to get the best of both worlds — i.e., a CNN easily recognizes spatial data and passes it to RNN for temporal data storing.

Step-by-Step Process of Time Series Forecasting Using Machine Learning.

With all the above said and done, let’s look at some common practices of forecasting using machine learning. Taking things step by step here is crucial for smooth, high-quality predictive time modeling and resulting forecasting.

Preparation Stage : Project goal definition — start with the comprehensive outline and understanding of minor and major milestones and goals. This is where preliminary research should be carried out to most properly tackle business area specifics.
Data gathering and exploration — continuing with thorough preparation, specific data types to be analyzed and processed must be settled. Data visualization charts and plot graphs can be used for this.
Data preparation and time series decomposition — specialized professionals should clean all involved data, gaining valuable insights and extracting proper variables. These variables can then be used for time series decomposition.

Modeling Stage : Forecasting models evaluation — based on all the preliminary research and prep data, different forecasting models are tested and evaluated to pick the most efficient one(s).
Forecasting model training and performance estimation — the picked machine learning algorithms for time series are then optimized through cross-validation and trained.

Testing Stage : Forecasting models run on testing data with known results — a step necessary for making sure the picked algorithms do their work properly.
Accuracy and performance optimization — the last phase of polishing up algorithms to achieve the best forecasting speed and accuracy.

Deployment Stage : Data transformation and visualization — to integrate the resulting forecasting model(s) with the production at hand, the gathered data must be conveniently transformed and visualized for further processing.
Forecasting models revision and improvements — time series forecasting is always iterative, meaning that multiple ongoing revisions and optimizations must be implemented to continuously improve the forecasting performance.

Go through the following [article](https://www.influxdata.com/time-series-forecasting-methods/) to understand about time series forecasting methods and models.

Also go through this [article](https://www.analyticsvidhya.com/blog/2020/11/stock-market-price-trend-prediction-using-time-series-forecasting/) to see how the model is implemented in python.

Go through the following [article](https://www.kdnuggets.com/2020/01/stock-market-forecasting-time-series-analysis.html) to understand how time series forecasting is implemented on a dataset by python.

The stock market is very unpredictable, any geopolitical change can impact the share trend of stocks in the share market, doing a  reliable trend analysis is very difficult. The most efficient way to solve this kind of issue is with the help of Machine learning and Deep learning.
A popular and widely used statistical method for time series forecasting is the ARIMA model, which is what we will use to predict stock market trends.

Watch this [video](https://youtu.be/Y2khrpVo6qI) to understand more about ARIMA model

In this [article](https://towardsdatascience.com/time-series-forecasting-predicting-stock-prices-using-an-arima-model-2e3b3080bd70) , the ARIMA model is shown in action and is being used to predict TESLA stock price 

You can watch this [playlist](https://youtube.com/playlist?list=PLqYFiz7NM_SMC4ZgXplbreXlRY4Jf4zBP) to know more about time series forecasting.
