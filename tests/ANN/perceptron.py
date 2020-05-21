
import random

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
                    print 'iterations:%s'% self.iteration
                    print 'weight1:%s\nweight2:%s'% (self.weights[0],self.weights[1])
                    if round(self.weights[0],2) >= 0:
                        print 'linear model: y='+str(round(self.weights[1],2))+'x+'+str(round(self.weights[0],2))
                    else:
                        print 'linear model: y='+str(round(self.weights[1],2))+'x'+str(round(self.weights[0],2))
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