from operator import matmul
from random import randint
from socket import SO_J1939_FILTER
import numpy as np
import math
import matplotlib.pyplot as plt

def activate(x,W,b): #sigmoid function
    res = np.add(np.matmul(W,x), b)
    return 1./(1+np.exp(-res))

    
def cost(W2,W3,W4,b2,b3,b4, x1, x2, y):
    costvec = np.zeros(10)
    for i in range(10):
        x = [x1(i), x2(i)]
        a2 = activate(x,W2,b2)
        a3 = activate(a2,W3,b3)
        a4 = activate(a3,W4,b4)
        print(a4,  [y1[i], y2[i]], np.linalg.norm(a4 - [y1[i], y2[i]], 2) )
        costvec[i] = np.linalg.norm(a4 - [y1[i], y2[i]], 2)
        
    return 0.1*math.pow(np.linalg.norm(costvec,2),2)


######### DATA ##########
x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7]
x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]
y1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y2 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# Initialize weights and biases
W2 = 0.5*np.random.random((2,2)); W3 = 0.5*np.random.random((3,2)); W4 = 0.5*np.random.random((2,3))
b2 = 0.5*np.random.random(2); b3 = 0.5*np.random.random(3); b4 = 0.5*np.random.random(2)
# Forward and Back propagate
eta = 0.05; # learning rate
Niter = 1000; # number of SG iterations
savecost = np.ones(Niter); # value of cost function at each iteration
for counter in range(1,Niter):
    k = randint(0,9); # choose a training point at random
    x = [x1[k], x2[k]]
    # Forward pass
    a2 = activate(x,W2,b2)
    a3 = activate(a2,W3,b3)
    a4 = activate(a3,W4,b4)
    # Backward pass
    delta4 = np.multiply(a4,np.subtract(1,a4),np.subtract (a4, [y1[k],y2[k]] ))
    delta3 = np.multiply(a3,np.subtract(1,a3),np.matmul(W4.T, delta4))
    delta2 = np.multiply(a2,np.subtract(1,a2),np.matmul(W3.T, delta3))
    # Gradient step
    W2 = W2 - eta*np.outer(delta2,x)
    W3 = W3 - eta*np.outer(delta3,a2.T)
    W4 = W4 - eta*np.outer(delta4,a3.T)
    b2 = b2 - eta*delta2
    b3 = b3 - eta*delta3
    b4 = b4 - eta*delta4
    #Monitor progress
    costvec = np.zeros(10)
    costvec = np.zeros(10)
    for i in range(10):
        x = [x1[i], x2[i]]
        a2 = activate(x,W2,b2)
        a3 = activate(a2,W3,b3)
        a4 = activate(a3,W4,b4)
        print(a4,  [y1[i], y2[i]], np.linalg.norm(a4 - [y1[i], y2[i]], 2) )
        costvec[i] = np.linalg.norm(a4 - [y1[i], y2[i]], 2)
        
    savecost [Niter-1] = 0.1*math.pow(np.linalg.norm(costvec,2),2)
    #print(savecost[Niter-1])

fig, ax = plt.subplots()
ax.plot(np.log(range(Niter)),savecost)
plt.show()
