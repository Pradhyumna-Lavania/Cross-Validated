from math import log
import numpy as np
import scipy.stats
from scipy.stats.stats import normaltest


tita = np.genfromtxt("titanic.csv", delimiter=',')

feat = np.array([0.0 for i in range(6)])

train = 50

## Function to normalize the features
def feat_proc(feat):
    feat[3]/= 28
    feat[4]/= 3
    feat[5]/= 2
    feat[6]/= 50
    return feat

## Normalizing Features
for i in range(1, len(tita)):
    tita[i] = np.array(feat_proc(tita[i]))

## The log-likelihood function (This would be maximized)
def loglike(feat):
    sum = 0
    for i in range(1, len(tita)-train):
        logo = scipy.stats.norm.cdf(np.dot(np.array([tita[i][j] for j in range(1,7)]), feat))
        #print("logo: ", logo)
        sum+= tita[i][0]* np.log(logo + 0.0001) + (1 - tita[i][0])*np.log(1 - logo + 0.0001)
    return sum

def summalog(feat):
    sum = 0
    for i in range(1, len(tita)):
        cdfo = scipy.stats.norm.cdf(np.dot(np.array([tita[i][j] for j in range(1,7)]), feat)) + 0.00001
        #print("CEFo: ", cdfo)
        sum+= (tita[i][0]-cdfo)/(cdfo*(1-cdfo))
    return sum

## Function to calculate partial derivative of loglike(feat)
def derivative_partial(independent, th, val):
    inc = list(feat)
    inc[int(independent)] = val
    step = list(inc)
    step[int(independent)] = step[int(independent)] + th
    derivative = (loglike(step) - loglike(inc))/th
    return derivative

## Function to find zeroes of log likelihood function by newtons method of iteration 
## (As loglike can have maximum value of zero hence maximizing loglike)
def newton(param):
    deri = derivative_partial(param, 0.1,0)
    fun = loglike(feat)
    frac = deri/fun

    n = 0
    n1 = n - frac

    par = list(feat)
    par[param] = n1

    while frac >= 0.1:
        deri = derivative_partial(param, 0.1, n1)
        fun = loglike(par)
        frac = deri/fun

        n = n1
        n1 = n - (frac)
        par[param] = n1
    return n1


#__main__

## Feature-wise applying newtons method

for i in range(len(feat)):
    feat[i] = newton(i)

print("MLE estimates of beta coefficients are : ",feat)


## Testing model
score = 0
for i in range(len(tita)-train, len(tita)):
    nomra = scipy.stats.norm.cdf(np.dot(np.array([tita[i][j] for j in range(1,7)]), feat))
    if tita[i][0] == 1 and nomra >= 0.5 :
        score+=1
    elif tita[i][0] == 0 and nomra <= 0.5:
        score+= 1
    
    """ if(tita[i][0]== 1):
        print("Expected : {} || Got : {}, Matched : {}".format(tita[i][0], nomra, nomra>=0.5)) """

print("Score is: {}/{}".format(score, train))
print("Hence accuracy is : {}%".format(score*100/train))

print("Log liklihood is: ", loglike(feat))

jack = np.array([0,1,1,20,0,0,7.5]) ## Here first parameter 0 is just dummy for correct array size to use feat_proc
jack = feat_proc(jack)

nomra = scipy.stats.norm.cdf(np.dot(np.array([jack[j] for j in range(1,7)]), feat))
print("Probablity of survival of Jack : ", nomra)

rose = np.array([0,1,0,19,1,1,512]) ## Here first parameter 0 is just dummy for correct array size to use feat_proc
rose = feat_proc(rose)

nomra = scipy.stats.norm.cdf(np.dot(np.array([rose[j] for j in range(1,7)]), feat))
print("Probablity of survival of Rose : ", nomra)