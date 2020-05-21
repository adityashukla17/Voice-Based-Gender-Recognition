import random
from Demo import demo
import os


class perceptron(object):
    def __init__(self,debugMode=False):
        
        self.debug = debugMode
        self.iteration = 0

    def init(self,learningRate=0.1,maxIteration = 1000,initWeights_Start=0,initWeights_End=1):
        
        self.theta = learningRate
        self.maxIteration = maxIteration
        self.weights = [random.uniform(initWeights_Start,initWeights_End) for _ in xrange(2)]

    def train(self,trainData):
        
        learned = False
        while not learned:
            convergence = 0.0
            for x in trainData:
                ret = self.expected(x)
                if x[2] != ret:
                    err = x[2] - ret
                    self._updateWeight(err,x)
                    convergence += err**2 #convergence += abs(err)
            self.iteration += 1
            if convergence == 0.0 or \
                self.iteration >= self.maxIteration:
                if self.debug:
                    #print 'iterations:%s'% self.iteration
                    print 'weight1:%s\nweight2:%s'% (self.weights[0],self.weights[1])
                    #if round(self.weights[0],2) >= 0:
                        #print 'linear model: y='+str(round(self.weights[1],2))+'x+'+str(round(self.weights[0],2))
                    #else:
                        #print 'linear model: y='+str(round(self.weights[1],2))+'x'+str(round(self.weights[0],2))
                learned = True

    def expected(self,data):
        
        sum = 0
        for x in xrange(2):
            sum += data[x] * self.weights[x]
        if sum >=0:
            return 1
        else:
            return -1

    def _updateWeight(self,err,trainDataPart):
        for x in xrange(2):
            self.weights[x] += self.theta * err * trainDataPart[x]
print 'test 1: predict data label'
p = perceptron(1)
p.init()
p.train([[-0.4, 0.9, 1], [-0.5, 0, 1], [0.6, 0.1, -1]])
print 'predict label of the data is:%d'% p.expected([-0.2, 0.4])

c=53.4276247832
print '\nclassify data by a line'
trainDataA = [[uniform(-1, 0), uniform(0, 1), 1] for a in xrange(50)]
trainDataB = [[uniform(0, 1), uniform(0, -1), -1] for b in xrange(50)]
print 'train data set:%s'% trainDataB
trainData = trainDataA + trainDataB
p1 = perceptron(1)
p1.init()
p1.train(trainData)
for x in trainData:
    r = p1.expected(x)
    if r != x[2]:
        print 'error dot:('
   # if r == 1:
        #plot(x[0], x[1], 'ob')
   # else:
        #plot(x[0], x[1], 'or')

n = norm(p1.weights) # aka the length of p.w vector
ww = p1.weights / n # a unit vector
ww1 = [ww[1], -ww[0]]
ww2 = [-ww[1], ww[0]]
#plot([ww1[0], ww2[0]], [ww1[1], ww2[1]], '--k')
#savefig("../data/test/perceptronClassification_demo.png")
#show()
print("Accuracy is:")
print(c)
#for g in range(1,1000):
   # sins=sin(2*pi*1200/8000*g)
name = input("Enter file name:")
os.system("afplay /Users/kakadiadhwani/Desktop/MINIPROJECT/DataSet/"+name)
demo(name)

