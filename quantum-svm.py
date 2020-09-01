import dimod
import neal
import numpy as np
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
import sklearn as sk
from sklearn import preprocessing
import sklearn.datasets as datasets
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import dwave.system.samplers as dw
from dwave.system import EmbeddingComposite

#Temporary, just for tests
def load_dataset_cancer(path,filename):
    strdataset = np.loadtxt(path+filename,dtype=str)
    num_features = len(strdataset[0].split(','))-2
    count = -1
    for data in strdataset:
        count+=1
        for feature in range(0,num_features+2):
            if data.split(',')[feature] == '?':
                strdataset = np.delete(strdataset,count)
                count+=-1
                break
    
    num_samples = len(strdataset)
    dataset = np.array([[0]*(num_features+1)]*num_samples)
    count = -1
    for data in strdataset:
        count+=1
        for feature in range(1,num_features+2):
            dataset[count][feature-1] = float(data.split(',')[feature])

    x = dataset[:,:-1]
    y = dataset[:,-1]

    return x, y

#This has to be changed so that it works both for the pulsar and cancer dataset
def load_dataset(path):
    dataset = np.loadtxt(path,dtype=float)
    x = dataset[:,:-1]
    y = dataset[:,-1]
    num_samples = len(x)
    num_features = len(x[0])
    return x, y, num_samples, num_features

def preprocess(x,y,sampling_strategy):
    undersample = RandomUnderSampler(sampling_strategy=sampling_strategy)
    x, y = undersample.fit_resample(x,y)
    num_samples = len(x)
    num_features = len(x[0])
    #preprocessing.scale() scales the data to values between 0 and 1
    x = sk.preprocessing.scale(x)
    #The targets have to be +1 or -1
    for target in range(0,num_samples):
        if y[target]==2: #for cancer dataset put a 2, for pulsar dataset put a 0. It has to be adapted so that it works for both
            y[target] = -1
        else:
            y[target] = 1
    return x, y, num_samples, num_features

def split_data(x,y,train_size):
    #Split the generated data into a training dataset and a test dataset
    xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=train_size)
    num_train_samples = len(xtrain)
    return xtrain, xtest, ytrain, ytest, num_train_samples

#Definition of the kernel, if gamma = -1, then it's the dot product (the linear kernel)
def kernel(xn,xm,gamma):
    if gamma == -1:
        dot = 0
        for feature in range(0,num_features):
            dot += xn[feature]*xm[feature]
        return dot
    xn = np.atleast_2d(xn)
    xm = np.atleast_2d(xm)
    return np.exp(-gamma * np.sum((xn[:,None] - xm[None,:])**2, axis=-1))

#Evaluate the model's performance with classification metrics
def evaluate(prediction):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for testpoint in range(0,num_samples-num_train_samples):
        if ytest[testpoint] == 1:
            if prediction[testpoint] == 1:
                tp+=1
            else:
                fn+=1
        else:
            if prediction[testpoint] == -1:
                tn+=1
            else:
                fp+=1
       
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    accuracy = (tp+tn)/(tp+tn+fp+fn)

    return precision, recall, f1, accuracy

#Calculate the ROC and the AUC
def eval_roc_curve(yscore,title):
    fpr, tpr, threshold = metrics.roc_curve(ytest,yscore)
    roc_auc = metrics.auc(fpr,tpr)
    plt.plot(fpr, tpr, 'k', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(title)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

#Save the classification metrics in a file
def save_files(path,filename,precision,recall,f1,accuracy):
    with open(path+filename,'w') as r:
        r.write('Precision: %f' %precision)
        r.write(', ')
        r.write('Recall: %f' %recall)
        r.write(', ')
        r.write('F1-score: %f' %f1)
        r.write(', ')
        r.write('Accuracy: %f' %accuracy)


class classicSVM:
    def __init__(self,xtrain,ytrain,xtest,C,kernel):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.model = SVC(kernel=kernel, C=C)
        self.model.fit(self.xtrain,self.ytrain)
    
    def get_yscore(self):
        yscore = self.model.decision_function(self.xtest)
        return yscore
    
    def get_predictions(self):
        predictions = self.model.predict(self.xtest)
        return predictions

class simulated_annealingSVM:
    def __init__(self,xtrain,ytrain,xtest,num_samples,num_train_samples,num_features,num_qubits_per_a,num_iter):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.num_samples = num_samples
        self.num_train_samples = num_train_samples
        self.num_features = num_features
        self.num_qubits_per_a = num_qubits_per_a
        #Definition of the quadratic coefficients for every pair of qubits, an(qi)*am(qj), as B^(i+j)*yn*ym*xn*xm
        #This corresponds to the first term of the classical svm minimisation problem summation(an*am*yn*ym*xn*xm)
        self.quadratic = {}
        for data in it.product(np.arange(0,self.num_train_samples),repeat=2):
            for qubit in it.product(np.arange(0,self.num_qubits_per_a),repeat=2):
                self.quadratic[("a{}(q{})".format(data[0],qubit[0]),"a{}(q{})".format(data[1],qubit[1]))] = 0.5*(2**(qubit[0]+qubit[1]))*self.ytrain[data[0]]*self.ytrain[data[1]]*kernel(self.xtrain[data[0]],self.xtrain[data[1]],16)
        #Definition of the linear coefficients, for each qubit an(qi), as -B^i
        #This corresponds to the second term of the classical svm minimisation problem -summation(an)
        self.linear = {}
        for data in range(0,self.num_train_samples):
            for qubit in range(0,self.num_qubits_per_a):
                self.linear["a{}(q{})".format(data,qubit)] = -(2**qubit)

        #Create the binary quadratic model and find the lowest energy solution with a simulated annealing sampler
        self.bqm = dimod.BinaryQuadraticModel(self.linear, self.quadratic, 0, 'BINARY')
        self.sampler = neal.SimulatedAnnealingSampler()
        self.sampleset = self.sampler.sample(self.bqm, num_reads=num_iter*10)
        self.sampleset_iterator = self.sampleset.samples(num_iter)

    #As I didn't add the squared penalty term xi*(summation(B^k*an(qk)*tn))^2, I have defined a function
    #which goes through the all the samples and checks if the constraint summation(an*yn)=0 is fulfilled
    def check_constraint(self,sample):
        #Calculation of the lagrange multipliers (alphas) with the binary encoding
        lagrange_multipliers = {}
        for data in range(0,self.num_train_samples):
            lagrange_multipliers["a{}".format(data)] = 0
            for qubit in range(0,self.num_qubits_per_a):
                lagrange_multipliers["a{}".format(data)] += self.sampleset_iterator[sample]["a{}(q{})".format(data,qubit)] * (2*qubit)

        #Calculation of the summation(an*yn)
        sum_lagrange_target = 0
        for data in range(0,self.num_train_samples):
            sum_lagrange_target += lagrange_multipliers["a{}".format(data)]*self.ytrain[data]
        
        #Check if the constraint is fulfilled
        if sum_lagrange_target==0:
            return True, lagrange_multipliers
        else:
            return False, lagrange_multipliers

    #In case there isn't any sample which fulfills the constraint summation(an*yn)=0, calculate the mean of
    #all the distinct samples found
    def mean_samples(self,sampleset,max_samples):
        #Save the sampleset as a pandas dataframe
        sample_df = sampleset.to_pandas_dataframe()
        #Take only the distinct samples
        unique_samples = sample_df.drop_duplicates()
        #Take only max_samples samples
        slice_unique_samples = unique_samples.iloc[:max_samples,:num_samples*self.num_qubits_per_a]
        #Calculate the mean of those samples
        mean_lagrange_multipliers = slice_unique_samples.mean()
        
        #Calculation of the lagrange multipliers (alphas) with the binary encoding
        lagrange_multipliers = {}
        for data in range(0,self.num_train_samples):
            lagrange_multipliers["a{}".format(data)] = 0
            for qubit in range(0,self.num_qubits_per_a):
                lagrange_multipliers["a{}".format(data)] += mean_lagrange_multipliers["a{}(q{})".format(data,qubit)] * (2**qubit)

        return lagrange_multipliers

    #Calculate the weights from the lagrange multipliers (alphas)
    def get_weights(self,alphas):
        weights = {}
        for feature in range(0,self.num_features):
            weights["w{}".format(feature)] = 0
            for data in range(0,self.num_train_samples):
                weights["w{}".format(feature)] += alphas["a{}".format(data)]*self.ytrain[data]*self.xtrain[data][feature]

        return weights

    def run_simulated_anneal(self):
        #Check the samples until one which fulfills the constraint is found. Then, calculate the weights
        count = 0
        #max_samples is the maximum number of samples to take to calculate the mean
        max_samples = 10
        valid = False
        while not valid and count<len(self.sampleset_iterator):
            valid, self.valid_lagrange_multipliers = self.check_constraint(count)
            count+=1
        if valid:
            self.valid_weights = self.get_weights(self.valid_lagrange_multipliers)
        #If there isn't any sample fulfilling the constraint, calculate the mean of the distinct samples
        else:
            print("Sample fulfilling constraint not found")
            self.valid_lagrange_multipliers = self.mean_samples(self.sampleset,max_samples)
            self.valid_weights = self.get_weights(self.valid_lagrange_multipliers)
        
    def get_yscore(self):
        self.run_simulated_anneal()
        self.yscore = []
        for testpoint in range(0,self.num_samples-self.num_train_samples):
            self.yscore.append(0)
            for feature in range(0,self.num_features):
                self.yscore[testpoint] += self.valid_weights["w{}".format(feature)]*self.xtest[testpoint][feature]
        return self.yscore
    
    def get_predictions(self):
        self.prediction = []
        for testpoint in range(0,self.num_samples-self.num_train_samples):
            if self.yscore[testpoint]>=0:
                self.prediction.append(1)
            else:
                self.prediction.append(-1)
        return self.prediction

class dwaveSVM:
    def __init__(self,xtrain,ytrain,xtest,num_samples,num_train_samples,num_features,num_qubits_per_a,num_iter,num_experiments):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.num_samples = num_samples
        self.num_train_samples = num_train_samples
        self.num_features = num_features
        self.num_qubits_per_a = num_qubits_per_a
        self.num_iter = num_iter
        self.num_experiments = num_experiments
        self.data_arr = np.arange(0,self.num_train_samples-1)
        np.random.shuffle(self.data_arr)
        self.num_samples_per_split = 9 #for cancer dataset put a 9, for pulsar dataset put a 14. It has to be adapted so that it works for both
        self.num_splits = int(self.num_train_samples/self.num_samples_per_split)
        self.data_split = np.split(self.data_arr,self.num_splits)

    def check_constraint(self,sample,split):
        #Calculation of the lagrange multipliers (alphas) with the binary encoding
        lagrange_multipliers = {}
        for data in self.data_split[split]:
            lagrange_multipliers["a{}".format(data)] = 0
            for qubit in range(0,self.num_qubits_per_a):
                lagrange_multipliers["a{}".format(data)] += self.sampleset_iterator[sample]["a{}(q{})".format(data,qubit)] * (2**qubit)

        #Calculation of the summation(an*yn)
        sum_lagrange_target = 0
        for data in self.data_split[split]:
            sum_lagrange_target += lagrange_multipliers["a{}".format(data)]*self.ytrain[data]
        
        #Check if the constraint is fulfilled
        if sum_lagrange_target==0:
            return True, lagrange_multipliers
        else:
            return False, lagrange_multipliers

    #In case there isn't any sample which fulfills the constraint summation(an*yn)=0, calculate the mean of
    #all the distinct samples found
    def mean_samples(self,sampleset,split,max_samples):
        #Save the sampleset as a pandas dataframe
        sample_df = sampleset.to_pandas_dataframe()
        #Take only the distinct samples
        unique_samples = sample_df.drop_duplicates()
        #Take only max_samples samples
        slice_unique_samples = unique_samples.iloc[:max_samples,:self.num_samples_per_split*self.num_qubits_per_a]
        #Calculate the mean of those samples
        mean_lagrange_multipliers = slice_unique_samples.mean()
        
        #Calculation of the lagrange multipliers (alphas) with the binary encoding
        lagrange_multipliers = {}
        for data in self.data_split[split]:
            lagrange_multipliers["a{}".format(data)] = 0
            for qubit in range(0,self.num_qubits_per_a):
                lagrange_multipliers["a{}".format(data)] += mean_lagrange_multipliers["a{}(q{})".format(data,qubit)] * (2**qubit)

        return lagrange_multipliers

    #Calculate the weights from the lagrange multipliers (alphas)
    def get_weights(self,alphas,split):
        weights = {}
        for feature in range(0,self.num_features):
            weights["w{}".format(feature)] = 0
            for data in self.data_split[split]:
                weights["w{}".format(feature)] += alphas["a{}".format(data)]*self.ytrain[data]*self.xtrain[data][feature]

        return weights

    #Predictions
    def predict(self,weights):
        prediction = []
        yscore = []
        for testpoint in range(0,self.num_samples-self.num_train_samples):
            yscore.append(0)
            for feature in range(0,self.num_features):
                yscore[testpoint] += weights["w{}".format(feature)]*self.xtest[testpoint][feature]
        
            if yscore[testpoint]>=0:
                prediction.append(1)
            else:
                prediction.append(-1)

        return prediction, yscore

    def run_dwave(self):
        self.predictions = {}
        self.yscore = {}
        for split in range(0,self.num_experiments):
        #Definition of the quadratic coefficients for every pair of qubits, an(qi)*am(qj), as B^(i+j)*yn*ym*xn*xm
        #This corresponds to the first term of the classical svm minimisation problem summation(an*am*yn*ym*xn*xm)
            self.quadratic = {}
            for data in it.product(self.data_split[split],repeat=2):
                for qubit in it.product(np.arange(0,self.num_qubits_per_a),repeat=2):
                    self.quadratic[("a{}(q{})".format(data[0],qubit[0]),"a{}(q{})".format(data[1],qubit[1]))] = float(0.5*(2**(qubit[0]+qubit[1]))*self.ytrain[data[0]]*self.ytrain[data[1]]*kernel(self.xtrain[data[0]],self.xtrain[data[1]],16))
        #Definition of the linear coefficients, for each qubit an(qi), as -B^i
        #This corresponds to the second term of the classical svm minimisation problem -summation(an)
            self.linear = {}
            for data in self.data_split[split]:
                for qubit in range(0,self.num_qubits_per_a):
                    self.linear["a{}(q{})".format(data,qubit)] = -(2**qubit)

            self.bqm = dimod.BinaryQuadraticModel(self.linear, self.quadratic, 0, 'BINARY')
            self.sampler = EmbeddingComposite(dw.DWaveSampler())
            self.sampleset = self.sampler.sample(self.bqm, num_reads=self.num_iter)
            self.sampleset_iterator = self.sampleset.samples(self.num_iter)

        #Check the samples until one which fulfills the constraint is found. Then, calculate the weights
            count = 0
            valid = False
            while not valid and count<len(self.sampleset_iterator):
                valid, self.valid_lagrange_multipliers = self.check_constraint(count,split)
                count+=1
            if valid:
                self.valid_weights = self.get_weights(self.valid_lagrange_multipliers,split)
                self.predictions["exp{}".format(split)], self.yscore["exp{}".format(split)] = self.predict(self.valid_weights)
        #If there isn't any sample fulfilling the constraint, calculate the mean of the distinct samples
            else:
                print("Sample fulfilling constraint not found")
                self.valid_lagrange_multipliers = self.mean_samples(self.sampleset,split,10)
                self.valid_weights = self.get_weights(self.valid_lagrange_multipliers,split)
                self.predictions["exp{}".format(split)], self.yscore["exp{}".format(split)] = self.predict(self.valid_weights)

    def get_yscore(self):
        self.run_dwave()
        self.yscore_mean = []
        for testpoint in range(0,self.num_samples-self.num_train_samples):
            yscore_sum = 0
            for split in range(0,self.num_experiments):
                yscore_sum+=self.yscore["exp{}".format(split)][testpoint]
            self.yscore_mean.append(yscore_sum/self.num_experiments)
        return self.yscore_mean
    
    def get_predictions(self):
        self.predict_mean = []
        for testpoint in range(0,self.num_samples-self.num_train_samples):
            if self.yscore_mean[testpoint]>=0:
                    self.predict_mean.append(1)
            else:
                self.predict_mean.append(-1)
        return self.predict_mean


path = 'C:/tmp/'
filename = 'breast-cancer-wisconsin.data'
x, y = load_dataset_cancer(path,filename) #load_dataset to be fixed
x, y, num_samples, num_features = preprocess(x,y,1)
xtrain, xtest, ytrain, ytest, num_train_samples = split_data(x,y,0.7)

#Instance an object for each of the classes, which are the distinct SVM models
classic_model = classicSVM(xtrain,ytrain,xtest,7,'rbf')
classic_yscore = classic_model.get_yscore()
classic_predictions = classic_model.get_predictions()
classic_precision, classic_recall, classic_f1, classic_accuracy = evaluate(classic_predictions)
eval_roc_curve(classic_yscore,'Classic ROC and AUC')
save_files(path,'classic-results.txt',classic_precision, classic_recall, classic_f1, classic_accuracy)

sim_model = simulated_annealingSVM(xtrain,ytrain,xtest,num_samples,num_train_samples,num_features,2,1000)
sim_yscore = sim_model.get_yscore()
sim_predictions = sim_model.get_predictions()
sim_precision, sim_recall, sim_f1, sim_accuracy = evaluate(sim_predictions)
eval_roc_curve(sim_yscore,'Simulated annealing ROC and AUC')
save_files(path,'sim-results.txt',sim_precision, sim_recall, sim_f1, sim_accuracy)

dwave_model = dwaveSVM(xtrain,ytrain,xtest,num_samples,num_train_samples,num_features,2,1000,5)
dwave_yscore = dwave_model.get_yscore()
dwave_predictions = dwave_model.get_predictions()
dwave_precision, dwave_recall, dwave_f1, dwave_accuracy = evaluate(dwave_predictions)
eval_roc_curve(dwave_yscore,'D-Wave ROC and AUC')
save_files(path,'dwave-results.txt',dwave_precision, dwave_recall, dwave_f1, dwave_accuracy)