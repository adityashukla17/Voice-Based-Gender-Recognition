import random
import math

class bpnn(object):
    """back propagation neural network"""

    def __init__(self,debugMode=False):
        
        self.debug = debugMode
        self.iteration = 0
        self.convergence =0.01

    def init(self,inputLayerNodeNum = 4,
        hiddenLayerNodeNum = 5,
        outputLayerNodeNum = 3,
        maxIteration = 1000,
        biasRange = [0,1],
        weightRange = [-1,1],
        learningRate = 0.15
        ):
        #constant
        self.inode = inputLayerNodeNum
        self.hnode = hiddenLayerNodeNum
        self.onode = outputLayerNodeNum
        self.maxIter = maxIteration
        self.lr = learningRate
        
        #initial bias
        self._init_bias(biasRange)

        #initial inputLayer-hiddenLayer and hiddenLayer-outputLayer weights
        self._init_Weights(weightRange)

        #save input and output data 
        self.hdata = [0] * self.hnode
        self.odata = [0] * self.onode

    def train(self,trainData):
        learned = False
        while not learned:
            for record in trainData:
                self.feedForward(record)
                error = self.backPropagate(record)
                if error<self.convergence:
                    learned = True
            
    def feedForward(self,data):
        for x in xrange(self.hnode):
            s = 0
            for t in xrange(len(data)-1):
                s += data[t] * self.hw[t][x] + self.hb[x]
            self.hdata[x] = [s,self._sigmoid(s)]

        for x in xrange(self.onode):
            s = 0
            for t in xrange(self.hnode):
                s += self.hdata[t][1] * self.ow[t][x] +self.ob[x]
            self.odata[x] = [s,self._sigmoid(s)]

    def backPropagate(self,data):
        o_deltas = [0] * self.onode
        h_deltas = [0] * self.hnode
        #update hiddenLayer-outputLayer weights
        for x in xrange(self.onode):
            o_deltas[x] = self.odata[x][1] * (1-self.odata[x][1]) * (data[-1] - self.odata[x][-1])
            for t in xrange(self.hnode):
                self.ow[t][x] += self.lr * self.hdata[t][1] * o_deltas[x]
                self.ob[x] += self.lr * o_deltas[x]
        #update inputLayer-hiddenLayer weights
        for x in xrange(self.hnode):
            s =0
            for p in xrange(self.onode):
                s += self.ow[x][p] * o_deltas[p]
            h_deltas[x] = self.hdata[x][1] * (1-self.hdata[x][1]) * s
            for t in xrange(self.inode):
                self.hw[t][x] += self.lr * data[t] * h_deltas[x]
                self.hb[x] +=self.lr * h_deltas[x]

        global_err = 0.0
        for x in xrange(self.onode):
            global_err += 0.5 * (data[-1]-self.odata[x][1])**2
        return global_err

    def _sigmoid(self,x):
        return 1.0/(1.0 + math.exp(-x))

    def _init_Weights(self,weightRange):
        self.hw = [[random.uniform(weightRange[0],weightRange[1]) \
        for t in xrange(self.hnode)]for x in xrange(self.inode)]
        self.ow = [[random.uniform(weightRange[0],weightRange[1]) \
        for t in xrange(self.onode)]for x in xrange(self.hnode)]

    def _init_bias(self,biasRange):
        self.hb = [random.uniform(biasRange[0],biasRange[1]) for x in xrange(self.hnode)]
        self.ob = [random.uniform(biasRange[0],biasRange[1]) for x in xrange(self.onode)]




trainData = []
with open('/Users/kakadiadhwani/Desktop/MINIPROJECT/tests/ANN/.data/iris.data') as data:
    line = data.readlines()
    for x in xrange(0,50):
        line[x] = line[x].replace('Iris-setosa','1')
        line[x] = line[x].replace('\n','')
        recordList = line[x].split(',')
        for y in xrange(len(recordList)):
            recordList[y] = float(recordList[y])
        trainData.append(recordList)
    for x in xrange(50,100):
        line[x] =line[x].replace('Iris-versicolor','2')
        line[x] = line[x].replace('\n','')
        recordList = line[x].split(',')
        for y in xrange(len(recordList)):
            recordList[y] = float(recordList[y])
        trainData.append(recordList)
    for x in xrange(100,150):
        line[x] =line[x].replace('Iris-virginica','3')
        line[x] = line[x].replace('\n','')
        recordList = line[x].split(',')
        for y in xrange(len(recordList)):
            recordList[y] = float(recordList[y])
        trainData.append(recordList)

#bp =bpnn()
#bp.init()
#bp.train(trainData)

