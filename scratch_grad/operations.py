import numpy as np


def add_forward(a, b):
    return a.data + b.data


def add_backward(a, b, gradient):
    a.grad += gradient
    b.grad += gradient
    # Ensure b.grad has the correct shape (32, 128)
    #b.grad += np.sum(gradient, axis=0, keepdims=True)



def sub_forward(a, b):
    # TODO
    return a.data - b.data
    
    


def sub_backward(a, b, gradient):
    # TODO
    a.grad += gradient
    b.grad += -gradient
    


def mul_forward(a, b):
    # TODO
    return a.data * b.data
    


def mul_backward(a, b, gradient):
    # TODO
    a.grad+=gradient*b.data
    b.grad+=gradient*a.data

    


def div_forward(a, b):
    # TODO
    return a.data / b.data
    


def div_backward(a, b, gradient):
    # TODO
    a.grad += gradient/b.data
    b.grad += gradient*(-a.data/b.data**2)
    


def matmul_forward(a, b):
    # TODO
    return np.dot(a.data, b.data)



def matmul_backward(a, b, gradient):
    # TODO
    a.grad += np.dot(gradient, np.transpose(b.data))
    b.grad += np.dot(np.transpose(a.data), gradient)


    

def mean_row_forward(a):
    return np.mean(a.data, axis=0)


def mean_row_backward(a, gradient):
    a.grad += np.expand_dims(gradient, axis=0) / a.data.shape[0]


def relu_forward(a):
    # TODO
    a.data = np.where(a.data <= 0, 0, a.data)
    return a.data  

    

def relu_backward(a, gradient):
    # TODO
    a.grad += (a.data > 0) * gradient


def sigmoid_forward(a):
    # TODO
    return 1/(1+np.exp(-a.data))


def sigmoid_backward(a, gradient):
    # TODO
    sigmoid = 1 / (1 + np.exp(-a.data))
    a.grad += gradient * sigmoid * (1 - sigmoid)
    


def log_forward(a):
    # TODO
    return np.log(a.data)


def log_backward(a, gradient):
    # TODO
    a.grad += gradient/a.data
    


def cross_entropy_forward(scores, label):
    # TODO
    exp_scores = np.exp(scores.data - np.max(scores.data))  
    prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(prob[np.arange(len(label.data)), label.data])
    return np.sum(correct_logprobs) / len(label.data)
    


def cross_entropy_backward(scores, label, gradient):
    # TODO
    exp_scores = np.exp(scores.data - np.max(scores.data))  
    prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    prob[np.arange(len(label.data)), label.data] -= 1
    scores.grad += (prob / len(label.data)) * gradient
    





