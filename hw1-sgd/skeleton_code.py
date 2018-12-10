import pandas as pd
import random
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


### Assignment Owner: Siac31

#######################################
#### Normalization


def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    # TODO
    """
    1. concate train and test along the axis 0
       new=np.concatenate((train,test),axis=0)

    2. drop the constant feature
       2.1 Compare each value to the corresponding value in the first row:

        In [46]: a == a[0,:]
        Out[46]: 
        array([[ True,  True,  True],
               [ True, False,  True],
               [ True, False,  True],
               [ True,  True,  True]], dtype=bool)
               
        2.2 A column shares a common value if all the values in that column are True:

        In [47]: indices=np.all(a == a[0,:], axis = 0)
        # or indices=(a==a[0,:]).all(axis=0)
        In [48]: indices
        Out[49]: array([ True, False,  True], dtype=bool)

        2.3 get arr using boolean indices
        In [50]: a=a[:,~indices]

        online code:
        a=a[:,~np.all(a == a[0,:],axis=0)]
        or
        a=a[:,~(a==a[0,:]).all(axis=0)]

     
       indices=(new == new[0,:]).all(axis = 0)
       new= new[:,~indices]

    3. find max and min and rescale the feature
        
       maximum=new.max(axis=0) 
       minimum=new.min(axis=0)

       interval=maximum-minimum

       new=(new-min)/interval
    4. split back along axis 0
       [train_normalized, test_normalized]= np.split(new,[train.shape[0]])


    """
    #1. concate train and test along the axis 0
    new=np.concatenate((train,test),axis=0)

    #2. drop the constant feature
    indices=(new == new[0,:]).all(axis = 0)
    new= new[:,~indices]

    #3. find max and min and rescale the feature
    maximum=new.max(axis=0) 
    minimum=new.min(axis=0)
    interval=maximum-minimum
    new=(new-minimum)/interval

    #4. split back along axis 0
    [train_normalized, test_normalized]= np.split(new,[train.shape[0]])

    return train_normalized, test_normalized




########################################
#### The square loss function

def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the square loss, scalar
    """
    #loss = 0 #initialize the square_loss
    #TODO
    
    #return np.linalg.norm(np.dot(X,theta.reshape((num_features,1)))-y,ord=2)**2/y.shape[0]
     num_instances=y.shape[0]
    return np.linalg.norm(np.dot(X,theta.reshape(X.shape[1]))-y,ord=2)**2/num_instances

########################################
### compute the gradient of square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO
    m=y.shape[0]
    xx = np.matmul(X.transpose(),X)
    return -2/m*(np.dot(xx,theta)-np.dot(X.transpose(),y))



###########################################
### Gradient Checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm.  Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions:
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1)

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by:
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error

    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO

#################################################
### Generic Gradient Checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. And check whether gradient_func(X, y, theta) returned
    the true gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    #TODO


####################################
#### Batch Gradient Descent
def batch_grad_descent(X, y, alpha=0.1, num_iter=1000, check_gradient=False):
    """
    In this question you will implement batch gradient descent to
    minimize the square loss objective

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run
        check_gradient - a boolean value indicating whether checking the gradient when updating
        
        stop criterion 10e-6??

    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features)
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.zeros(num_features) #initialize theta
    #TODO
    #solid alpha
    """
    for num in range(1,num_iter):
        loss_gradient = compute_square_loss_gradient(X, y, theta_hist[num-1])
        if np.linalg.norm(loss_gradient,ord=2) <= 10e-6:
            break
        theta_hist[num] =theta_hist[num-1] + alpha*loss_gradient
        loss_hist[num] = compute_square_loss(X, y, theta_hist[num])

    return theta_hist, loss_hist
    """
####################################
###Q2.4b: Implement backtracking line search in batch_gradient_descent
###Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
#TODO
   # backtracking line search
    def backtracking_step_size(func_current, gradient, beta, gamma, X, y, theta_current):
        func_derivative = - beta*np.dot(gradient.transpose(),gradient)
        func_new=compute_square_loss(X,y,theta_current+gradient)
        t=1
        while func_new > func_current + t*func_derivative:
            t=gamma*t
            func_new = compute_square_loss(X,y,theta_current+t*gradient)
        #return theta_new, func_new

        return theta_current+t*gradient, func_new




    
    loss_hist[0] = compute_square_loss(X,y,theta)
    beta=0.25
    gamma=0.25
    #beta = random.random()
    #sigma = random.uniform(0, .5)
    for num in range(1,num_iter):
        loss_gradient = compute_square_loss_gradient(X, y, theta_hist[num-1,:])
        if np.linalg.norm(loss_gradient,ord=2) <= 10e-4:
            break
        theta_hist[num], loss_hist[num] = backtracking_step_size(loss_hist[num-1], \
            loss_gradient, beta, gamma, X, y, theta_hist[num-1,:])
        #c=0.5
        #gamma=0.5
        #delta = c*loss_gradient 
        #loss_hist[num] = compute_square_loss(X, y, theta_hist[num-1,:] + loss_gradient)
        #t=1
        #while loss_hist[num] > loss_hist[num-1]+t*delta:
        #    t=gamma*t
        #    theta_hist[num]=theta_hist[num-1,:] + t*loss_gradient
        #    loss_hist[num]= compute_square_loss(X, y, theta_hist[num,:])
    theta_hist = theta_hist[0:num,:]
    loss_hist = loss_hist[0:num]
    return theta_hist, loss_hist
        

   

    

###################################################
### Compute the gradient of Regularized Batch Gradient Descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO
    m=y.shape[0]
    xx = np.matmul(X.transpose(),X)
    return -2/m*(np.dot(xx,theta)-np.dot(X.transpose(),y))-lambda_reg*2*theta 

###################################################
### Batch Gradient Descent with regularization term
def regularized_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features)
        loss_hist - the history of loss function without the regularization term, 1D numpy array.
    """
    (num_instances, num_features) = X.shape
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #Initialize loss_hist
    #TODO

#############################################
## Visualization of Regularized Batch Gradient Descent
##X-axis: log(lambda_reg)
##Y-axis: square_loss
    def backtracking_step_size(func_current, gradient, beta, gamma, X, y, theta_current, lambda_reg):
        func_derivative = - beta*np.linalg.norm(gradient)
        func_new=compute_square_loss(X,y,theta_current+gradient) +\
         lambda_reg*np.linalg.norm(theta_current+gradient,ord=2)**2
        t=1
        while func_new > func_current + t*func_derivative:
            t=gamma*t
            func_new = compute_square_loss(X,y,theta_current+t*gradient)+\
            lambda_reg*np.linalg.norm(theta_current+t*gradient,ord=2)
        #return theta_new, func_new

        return theta_current+t*gradient, func_new

    loss_hist[0] = compute_square_loss(X,y,theta)+lambda_reg*np.linalg.norm(theta,ord=2)
    beta=0.25
    gamma=0.25
    #beta = random.random()
    #sigma = random.uniform(0, .5)
    for num in range(1,num_iter):
        loss_gradient = compute_regularized_square_loss_gradient(X, y, theta_hist[num-1,:], lambda_reg)
        if np.linalg.norm(loss_gradient,ord=2) <= 10e-2:
            break
        theta_hist[num], loss_hist[num] = backtracking_step_size(loss_hist[num-1], \
            loss_gradient, beta, gamma, X, y, theta_hist[num-1,:], lambda_reg)
    theta_hist = theta_hist[0:num,:]
    loss_hist = loss_hist[0:num]
    return theta_hist, loss_hist
        

   

    

###################################################

#############################################
### Stochastic Gradient Descent
def stochastic_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    In this question you will implement stochastic gradient descent with a regularization term

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. 
                Usually it's set to 1/sqrt(t) or 1/t, where t means the number of iterrations
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features)
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    (num_instances, num_features) = X.shape
    theta = np.ones(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #Initialize loss_hist
    #TODO

    def backtracking_step_size(func_current, gradient, beta, gamma, X, y, theta_current, lambda_reg):
        func_derivative = - beta*np.linalg.norm(gradient)
        func_new=compute_square_loss(X,y,theta_current+gradient) +\
        lambda_reg*np.linalg.norm(theta_current+gradient,ord=2)**2
        t=1
        while func_new > func_current + t*func_derivative:
            t=gamma*t
            func_new = compute_square_loss(X,y,theta_current+t*gradient)+\
            lambda_reg*np.linalg.norm(theta_current+t*gradient,ord=2)
        #return theta_new, func_new

        return theta_current+t*gradient, func_new

    loss_hist[0] = compute_square_loss(X,y,theta)+lambda_reg*np.linalg.norm(theta,ord=2)
    beta=0.25
    gamma=0.25

    for num in range(1,num_iter):
        random_i = np.random.randint(num_instances,size=1)
        loss_gradient = -2*(np.dot(X[random_i],theta_hist[num-1])-y[random_i])*X[random_i]\
        -2*lambda_reg*theta_hist[num-1] # (num_features,1)
   
        if np.linalg.norm(loss_gradient,ord=2) <= 10e-6:
            break
        theta_hist[num], loss_hist[num] = backtracking_step_size(loss_hist[num-1], \
            loss_gradient, beta, gamma, X, y, theta_hist[num-1], lambda_reg)
     
    
    theta_hist = theta_hist[0:num]
    loss_hist = loss_hist[0:num] 
    return theta_hist, loss_hist
################################################
### Visualization that compares the convergence speed of batch
###and stochastic gradient descent for various approaches to step_size
##X-axis: Step number (for gradient descent) or Epoch (for SGD)
##Y-axis: log(objective_function_value) and/or objective_function_value

def main():
    #Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    B=1
    X_train = np.hstack((X_train, B*np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, B*np.ones((X_test.shape[0], 1)))) # Add bias term

    # TODO
    """
    theta_history, loss_history=batch_grad_descent(X_train, y_train, alpha=0.1, num_iter=1000, check_gradient=True)
    theta_history, loss_history=regularized_grad_descent(X_train, y_train, alpha=0.1, lambda_reg=1, num_iter=1000)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) 
    ax.scatter(np.arange(loss_history.shape[0]), loss_history, marker='o', linewidths=1, label='Armijo Rule')
    ax.set_xlabel('itteration')
    ax.set_ylabel('loss')
    ax.legend()
    plt.savefig('lossdescent.png')
    plt.savefig('lossdescent1.png')

    #my_path = os.path.abspath(__file__)
    #print(my_path)
    """
    # find a best lamda 

    lamda=np.array([1e-7,1e-5,1e-3,0.1,1,10,100])
    xticks=np.ones_like(lamda)
    for item in range(0,lamda.shape[0]):
        xticks[item]=xticks[item]+item

    test_loss=np.zeros_like(lamda)
    train_loss=np.zeros_like(lamda)
    for item in range(0,lamda.shape[0]):
        theta_history, loss_history=regularized_grad_descent(X_train, y_train, alpha=0.1, lambda_reg=lamda[item], num_iter=1000)
        train_loss[item] = loss_history[-1]
        test_loss[item] = compute_square_loss(X_test, y_test, theta_history[-1,:])

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.scatter(xticks, train_loss, color="red", marker='+', label='train loss')
    ax.scatter(xticks, test_loss, color="red", marker='o', label='test loss')
    ax.set_xticks(xticks)
    ax.set_xticklabels(['$10^{-7}$', '$10^{-5}$', '$10^{-3}$', '$0.1$', '$1$','$10$','$100$'], 
                   fontsize=18)
    ax.legend()
    plt.savefig('compare_loss.png')

if __name__ == "__main__":
    main()
