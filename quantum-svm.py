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
import dwave.inspector

#Load the dataset from a path in your computer, and divide it into the x and y
#according to the columns that belong to the data points and the column of labels
def load_dataset(path,filename,first_feature_col,last_feature_col,label_col):
    data_frame = pd.read_csv(path+filename,header=None,na_values='?')
    #Remove those samples that have na values '?'
    data_frame.dropna(inplace=True)
    #Choose the columns that correspond to the features of x
    x = data_frame.iloc[:,first_feature_col:last_feature_col].to_numpy()
    #Choose the column that corresponds to the labels
    y = data_frame.iloc[:,label_col].to_numpy()
    return x, y

#Make a preprocessing of data
def preprocess(x,y,sampling_strategy,wrong_target):
    #In case that the dataset is imbalanced, you can undersample it with some sampling strategy.
    #If you set sampling_strategy=0.5, it will mean that the ratio between the classes is 1:2.
    #If the dataset isn't imbalanced, the sampling_strategy can be equal to 1
    undersample = RandomUnderSampler(sampling_strategy=sampling_strategy)
    x, y = undersample.fit_resample(x,y)
    num_samples = len(x)
    num_features = len(x[0])
    #preprocessing.scale() scales the data to values between 0 and 1
    x = sk.preprocessing.scale(x)
    #The targets have to be +1 or -1, so if the dataset has different values as labels, choose
    #one of the labels as wrong_target and it will be changed to -1, while the other will be 1.
    #If the lables are already +1 and -1, set wrong_target=-1
    for target in range(0,num_samples):
        if y[target]==wrong_target:
            y[target] = -1
        else:
            y[target] = 1
    return x, y, num_samples, num_features

#Split the generated data into a training dataset and a test dataset
def split_data(x,y,train_size):
    xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=train_size)
    num_train_samples = len(xtrain)
    return xtrain, xtest, ytrain, ytest, num_train_samples

#Definition of the kernel
def kernel(xn,xm,gamma):
    #If gamma = -1, then it's the dot product (the linear kernel)
    if gamma == -1:
        dot = 0
        for feature in range(0,num_features):
            dot += xn[feature]*xm[feature]
        return dot
    #For different gammas, which are positive, the kernel is the usual RBF
    xn = np.atleast_2d(xn)
    xm = np.atleast_2d(xm)
    return np.exp(-gamma * np.sum((xn[:,None] - xm[None,:])**2, axis=-1))

#Evaluate the model's performance with classification metrics
def evaluate(prediction):
    tp = 0 #True positives
    fp = 0 #False positives
    tn = 0 #True negatives
    fn = 0 #False negatives
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

#Calculate the ROC and the AUC and plot it. It can be saved from the matplotlib window that opens
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

#Definition of the class of a classical support vector machine
class classicSVM:
    #Create model and train it with the training data
    def __init__(self,xtrain,ytrain,xtest,C,kernel):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.model = SVC(kernel=kernel, C=C)
        self.model.fit(self.xtrain,self.ytrain)
    
    #Obtain the y-score of the model
    def get_yscore(self):
        yscore = self.model.decision_function(self.xtest)
        return yscore
    
    #Obtain the predictions of the model
    def get_predictions(self):
        predictions = self.model.predict(self.xtest)
        return predictions

#Definition of the class of a quantum support vector machine that uses the simulated annealing sampler
class simulated_annealingSVM:
    #Define coefficients, create the binary quadratic model and find low-energy samples
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

        #Create the binary quadratic model and find the low-energy solutions with a simulated annealing sampler
        self.bqm = dimod.BinaryQuadraticModel(self.linear, self.quadratic, 0, 'BINARY')
        self.sampler = neal.SimulatedAnnealingSampler()
        self.sampleset = self.sampler.sample(self.bqm, num_reads=num_iter*10)
        self.sampleset_iterator = self.sampleset.samples(num_iter)

    #As I didn't add the squared penalty term xi*(summation(B^k*an(qk)*tn))^2, I have defined a function
    #which goes through all the samples and checks if the constraint summation(an*yn)=0 is fulfilled
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
        slice_unique_samples = unique_samples.iloc[:max_samples,:self.num_samples*self.num_qubits_per_a]
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

    #Check the samples until one which fulfills the constraint is found. Then, calculate the weights
    def run_simulated_anneal(self):
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
        
    #Obtain the y-score of the model
    def get_yscore(self):
        self.run_simulated_anneal()
        self.yscore = []
        for testpoint in range(0,self.num_samples-self.num_train_samples):
            self.yscore.append(0)
            for feature in range(0,self.num_features):
                self.yscore[testpoint] += self.valid_weights["w{}".format(feature)]*self.xtest[testpoint][feature]
        return self.yscore
    
    #Obtain the predictions of the model
    def get_predictions(self):
        self.prediction = []
        for testpoint in range(0,self.num_samples-self.num_train_samples):
            if self.yscore[testpoint]>=0:
                self.prediction.append(1)
            else:
                self.prediction.append(-1)
        return self.prediction

#Definition of the class of a quantum support vector machine that is executed in D-Wave's annealer
class dwaveSVM:
    def __init__(self,xtrain,ytrain,xtest,num_samples,num_train_samples,num_features,num_qubits_per_a,num_iter,num_experiments,num_samples_per_split):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.num_samples = num_samples
        self.num_train_samples = num_train_samples
        self.num_features = num_features
        self.num_qubits_per_a = num_qubits_per_a
        self.num_iter = num_iter
        self.num_experiments = num_experiments
        #Split the data in different parts that will be used for separate experiments.
        #Then, the average of their results will be calculated at the end
        self.data_arr = np.arange(0,self.num_train_samples-1)
        np.random.shuffle(self.data_arr)
        self.num_samples_per_split = num_samples_per_split
        self.num_splits = int(self.num_train_samples/self.num_samples_per_split)
        self.data_split = np.split(self.data_arr,self.num_splits)

    #As I didn't add the squared penalty term xi*(summation(B^k*an(qk)*tn))^2, I have defined a function
    #which goes through all the samples and checks if the constraint summation(an*yn)=0 is fulfilled
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

    #Obtain the y-score and the predictions of the model for one of the experiments that used one split of the data
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

    #Define coefficients, bqm, find low-energy solutions, check the constraint, calculate the weights and
    #calculate the y-score and predictions for each of the splits used in each experiment
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

            #Create the binary quadratic model and find the low-energy solutions with the D-Wave sampler
            self.bqm = dimod.BinaryQuadraticModel(self.linear, self.quadratic, 0, 'BINARY')
            self.sampler = EmbeddingComposite(dw.DWaveSampler())
            self.sampleset = self.sampler.sample(self.bqm, num_reads=self.num_iter)
            dwave.inspector.show(self.sampleset)
            self.sampleset_iterator = self.sampleset.samples(self.num_iter)

            #Check the samples until one which fulfills the constraint is found.
            #Then, calculate the weights, the y-score and the predictions
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

    #Obtain the y-score of the model, which is the mean of the y-scores of each of the experiments
    def get_yscore(self):
        self.run_dwave()
        self.yscore_mean = []
        for testpoint in range(0,self.num_samples-self.num_train_samples):
            yscore_sum = 0
            for split in range(0,self.num_experiments):
                yscore_sum+=self.yscore["exp{}".format(split)][testpoint]
            self.yscore_mean.append(yscore_sum/self.num_experiments)
        return self.yscore_mean
    
    #Obtain the predictions of the model from the average y-score
    def get_predictions(self):
        self.predict_mean = []
        for testpoint in range(0,self.num_samples-self.num_train_samples):
            if self.yscore_mean[testpoint]>=0:
                    self.predict_mean.append(1)
            else:
                self.predict_mean.append(-1)
        return self.predict_mean

#Load a dataset, preprocess the data and split it into training and testing
path = 'C:/tmp/'
filename = 'breast-cancer-wisconsin.data'
x, y = load_dataset(path,filename,1,9,10)
x, y, num_samples, num_features = preprocess(x,y,1,2)
xtrain, xtest, ytrain, ytest, num_train_samples = split_data(x,y,0.7)

#Instance an object for each of the classes, which are the distinct SVM models, and use the evaluate() 
#and the eval_roc_curve() functions to obtain the classification metrics and then save them with save_files()
#classic_model = classicSVM(xtrain,ytrain,xtest,7,'rbf')
#classic_yscore = classic_model.get_yscore()
#classic_predictions = classic_model.get_predictions()
#classic_precision, classic_recall, classic_f1, classic_accuracy = evaluate(classic_predictions)
#eval_roc_curve(classic_yscore,'Classic ROC and AUC')
#save_files(path,'classic-results.txt',classic_precision, classic_recall, classic_f1, classic_accuracy)

#sim_model = simulated_annealingSVM(xtrain,ytrain,xtest,num_samples,num_train_samples,num_features,2,1000)
#sim_yscore = sim_model.get_yscore()
#sim_predictions = sim_model.get_predictions()
#sim_precision, sim_recall, sim_f1, sim_accuracy = evaluate(sim_predictions)
#eval_roc_curve(sim_yscore,'Simulated annealing ROC and AUC')
#save_files(path,'sim-results.txt',sim_precision, sim_recall, sim_f1, sim_accuracy)

dwave_model = dwaveSVM(xtrain,ytrain,xtest,num_samples,num_train_samples,num_features,2,1000,5,9)
dwave_yscore = dwave_model.get_yscore()
dwave_predictions = dwave_model.get_predictions()
dwave_precision, dwave_recall, dwave_f1, dwave_accuracy = evaluate(dwave_predictions)
eval_roc_curve(dwave_yscore,'D-Wave ROC and AUC')
save_files(path,'dwave-results.txt',dwave_precision, dwave_recall, dwave_f1, dwave_accuracy)