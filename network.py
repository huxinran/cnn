import numpy as np
from layer import FullyConnectedLayer as FC
from utils import softmax
from utils import plot
import time
import matplotlib.pyplot as plt 

class NeuralNet:
    '''
    Neural Net class consists of k fully connected layers which takes an 
    array of data of size [din] and classify them into [dout] labels

    '''
    def __init__(self, dim):    
        '''
        dim  : an array of dimensions
        [din,       dh1, dh2, ..., dhk, dout]
         input -> | hidden layers    | -> output
        '''
        self.dim = dim
        self.l = len(self.dim) - 1
        
        self.w = [] 

        for i in range(self.l):
            self.w.append(FC.initWeight(self.dim[i], self.dim[i + 1]))
        
    def __repr__(self):
        return 'NeuralNet with {0.din} to {0.dhidden} to {0.dout}'.format(self)

    def predict(self, data):
        """
        given a input data, 
        return an output value and a list of data for each of the hidden layers
        """
        inputHidden = [None] * self.l
        inputHidden[0] = data
        
        for i in range(self.l - 1):
            inputHidden[i + 1] = np.maximum(0, FC.predict(inputHidden[i], self.w[i]))
        
        output = FC.predict(inputHidden[-1], self.w[-1])
        return output, inputHidden

    def computeLoss(self, output, label):
        """
        given the output and true label, 
        return softmax cross entropy loss and gradient on output
        """
        prob = softmax(output)
        rIdx = np.arange(label.shape[0])
        loss = -np.log(np.maximum(np.exp(-10), prob[rIdx, label]))
        dOutput = prob
        dOutput[rIdx, label] -= 1
        return loss, dOutput

    def gradient(self, dOutput, inputHidden):
        """
        given the graident on output and data for hidden layer
        computer the gradient for hidden layer weights
        """
        dwHidden = [None] * self.l
        for i in reversed(range(self.l)):
            dOutput, dwHidden[i] = FC.gradient(dOutput, inputHidden[i], self.w[i])      
            dOutput = dOutput * (1 * (inputHidden[i] > 0))

        # normalized the gradient by number of observations
        for dw in dwHidden:
            dw /= dOutput.shape[0]

        return dwHidden

    def applyGraident(self, dw, stepSize=0.001, regularization=0):
        """
        given gradient, apply gradient on weight with stepsize adjustment and 
        regularization
        """
        for i in range(self.l):
            self.w[i] -= (dw[i] * stepSize + self.w[i] * regularization)



    def trainIteration(self, data, label, debug=False):
        """
        one iteration of learning
        """
        # forawrd feed, get y and x_hidden
        output, inputHidden = self.predict(data)
        
        # measure loss and gradient on y
        loss, dOutput = self.computeLoss(output, label)
        
        # backprop, get gradient on weight
        dwHidden = self.gradient(dOutput, inputHidden)
        
        if debug:
            print('w = ', self.w)
            print("xhidden = ", inputHidden)
            print("y = ", output)
            print("dy = ", dOutput)
            print("dw = ", dwHidden)

        return output, loss, dwHidden

    def trainTestSplit(self, data, label, testPct=0.0):
        N = data.shape[0]
        trainSize = np.ceil(N * (1 - testPct)).astype(int)
        return data[0:trainSize,:], label[0:trainSize], data[trainSize:,:], label[trainSize:]

    def normalize(self, data):
        data = data - np.mean(data, axis=0)
        return data




    def normalize(self, data):
        data = data - np.mean(data, axis=0)
        return data


    def train(self, data, label, iteration, stepSize=0.001, regularization=0.0, testPct=0.0, debug=False):
        sampleSize = 10
        sampleIndex = np.random.choice(100, sampleSize, replace=False)
        start = time.time()
        
        for i in range(sampleSize):
            plt.subplot(sampleSize, 2, i * 2 + 1)
            plt.imshow(data[sampleIndex[i],:].reshape(28,28), cmap='gray')
            plt.pause(0.001)


        data = self.normalize(data)
        dTrain, lTrain, dTest, lTest = self.trainTestSplit(data, label, testPct)
        for t in range(iteration):
                    

            # computer gradient on weight
            outputTrain, lossTrain, dw = self.trainIteration(dTrain, lTrain, debug)
            
            # apply gradient on weight
            self.applyGraident(dw, stepSize, regularization)

            # all book keeping
            outputTest, _ = self.predict(dTest)

            avgLossTrain = np.mean(lossTrain)

            errRateTrain = np.mean(1 * (np.argmax(outputTrain, axis=1) != lTrain))
            errRateTest  = np.mean(1 * (np.argmax(outputTest, axis=1) != lTest))

            timeRemain = (time.time() - start) / (t + 1) * (iteration - t - 1)
            debugStr = 'Iter:{0:4d}|Time:{1:4.4f}|TrainErr:{2:4.4f}|Test Err:{3:4.4f}|Loss:{4:4.4f}'.format(t, timeRemain, errRateTrain,errRateTest,avgLossTrain)
            print(debugStr, end='\r')
            
            prob = softmax(outputTrain)
            for i in range(sampleSize):
                plt.subplot(sampleSize, 2, i * 2 + 2)
                plt.cla()
                plt.bar(np.arange(10), prob[sampleIndex[i],:])
                plt.ylim(-0.2, 1.2)
                plt.pause(0.001)
            
        print('\n\nTime total : {0}'.format(time.time() - start))


    def show(self):
        """
        visualized the decision bountary
        """
        x = np.linspace(-4, 4, 128)
        y = np.linspace(-4, 4, 128)
        data = []
        for xi in x:
            for yi in y:
                data.append([xi, yi])
        
        data = np.array(data)
        y, _ = self.predict(data)
        plot(data[:,0], data[:,1], np.argmax(y, axis=1))