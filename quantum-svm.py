import dimod
import neal
import numpy as np
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn import preprocessing
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import dwave.system.samplers as dw
from dwave.system import EmbeddingComposite

#Definition of load_dataset_cancer function for breast-cancer-wisconsin dataset
def load_dataset_cancer(path):
    strdataset = np.loadtxt(path,dtype=str)
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

    return dataset[:,:-1], dataset[:,-1]

#Definition of load_dataset function for pulsars dataset
def load_dataset(path):
    strdataset = np.loadtxt(path,dtype=str)
    num_samples = len(strdataset)
    num_features = len(strdataset[0].split(','))
    dataset = np.array([[0]*num_features]*num_samples)
    count = -1
    for data in strdataset:
        count+=1
        for feature in range(0,num_features):
            dataset[count][feature] = float(data.split(',')[feature])
    return dataset[:,:-1], dataset[:,-1]

#x, y = load_dataset_cancer('')
x, y = load_dataset('')
undersample = RandomUnderSampler(sampling_strategy=0.5)
x, y = undersample.fit_resample(x,y)
num_samples = len(x)
num_features = len(x[0])
#preprocessing.scale() scales the data to values between 0 and 1
x = preprocessing.scale(x)
#The targets have to be +1 or -1
for target in range(0,num_samples):
    if y[target]==0: #for cancer dataset put a 2, for pulsar dataset put a 0
        y[target] = -1
    else:
        y[target] = 1
#Split the generated data into a training dataset and a test dataset
train_size = 0.1
xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=train_size)
num_train_samples = int(num_samples*train_size)

#num_qubits_per_a is the number of qubits used to encode each lagrange multiplier (alphas)
num_qubits_per_a = 2

#Definition of the kernel, if gamma = -1, then it's the dot product (the linear kernel)
def kernel(xn, xm, gamma):
    if gamma == -1:
        dot = 0
        for feature in range(0,num_features):
            dot += xn[feature]*xm[feature]
        return dot
    xn = np.atleast_2d(xn)
    xm = np.atleast_2d(xm)
    return np.exp(-gamma * np.sum((xn[:,None] - xm[None,:])**2, axis=-1))

#The nomenclature used is an(qk), where n is the subindex for the lagrange multiplier and k is the 
#subindex for the qubits used to encode each alpha

B = 2 #B is the base used for encoding, it could be either 2 or 10

#ANNEALING
#Definition of the quadratic coefficients for every pair of qubits, an(qi)*am(qj), as B^(i+j)*yn*ym*xn*xm
#This corresponds to the first term of the classical svm minimisation problem summation(an*am*yn*ym*xn*xm)
sim_quadratic = {}
for data in it.product(np.arange(0,num_train_samples),repeat=2):
    for qubit in it.product(np.arange(0,num_qubits_per_a),repeat=2):
        sim_quadratic[("a{}(q{})".format(data[0],qubit[0]),"a{}(q{})".format(data[1],qubit[1]))] = 0.5*(B**(qubit[0]+qubit[1]))*ytrain[data[0]]*ytrain[data[1]]*kernel(xtrain[data[0]],xtrain[data[1]],16)
#Definition of the linear coefficients, for each qubit an(qi), as -B^i
#This corresponds to the second term of the classical svm minimisation problem -summation(an)
sim_linear = {}
for data in range(0,num_train_samples):
    for qubit in range(0,num_qubits_per_a):
        sim_linear["a{}(q{})".format(data,qubit)] = -(B**qubit)

#Create the binary quadratic model and find the lowest energy solution with a simulated annealing sampler
sim_bqm = dimod.BinaryQuadraticModel(sim_linear, sim_quadratic, 0, 'BINARY')
sim_sampler = neal.SimulatedAnnealingSampler()
num_iter = int(1000)
sim_sampleset = sim_sampler.sample(sim_bqm, num_reads=num_iter*10)
sim_sampleset_iterator = sim_sampleset.samples(num_iter)

#As I didn't add the squared penalty term xi*(summation(B^k*an(qk)*tn))^2, I have defined a function
#which goes through the all the samples and checks if the constraint summation(an*yn)=0 is fulfilled
def sim_check_constraint(sample):
    #Calculation of the lagrange multipliers (alphas) with the binary encoding
    lagrange_multipliers = {}
    for data in range(0,num_train_samples):
        lagrange_multipliers["a{}".format(data)] = 0
        for qubit in range(0,num_qubits_per_a):
            lagrange_multipliers["a{}".format(data)] += sim_sampleset_iterator[sample]["a{}(q{})".format(data,qubit)] * (B**qubit)

    #Calculation of the summation(an*yn)
    sum_lagrange_target = 0
    for data in range(0,num_train_samples):
        sum_lagrange_target += lagrange_multipliers["a{}".format(data)]*ytrain[data]
    
    #Check if the constraint is fulfilled
    if sum_lagrange_target==0:
        return True, lagrange_multipliers
    else:
        return False, lagrange_multipliers

#In case there isn't any sample which fulfills the constraint summation(an*yn)=0, calculate the mean of
#all the distinct samples found
def sim_mean_samples(sampleset):
    #max_samples is the maximum number of samples to take to calculate the mean
    max_samples = 10
    #Save the sampleset as a pandas dataframe
    sample_df = sampleset.to_pandas_dataframe()
    #Take only the distinct samples
    unique_samples = sample_df.drop_duplicates()
    #Take only max_samples samples
    slice_unique_samples = unique_samples.iloc[:max_samples,:num_samples*num_qubits_per_a]
    #Calculate the mean of those samples
    mean_lagrange_multipliers = slice_unique_samples.mean()
    
    #Calculation of the lagrange multipliers (alphas) with the binary encoding
    lagrange_multipliers = {}
    for data in range(0,num_train_samples):
        lagrange_multipliers["a{}".format(data)] = 0
        for qubit in range(0,num_qubits_per_a):
            lagrange_multipliers["a{}".format(data)] += mean_lagrange_multipliers["a{}(q{})".format(data,qubit)] * (B**qubit)

    return lagrange_multipliers

#Calculate the weights from the lagrange multipliers (alphas)
def sim_calculate_weights(alphas):
    weights = {}
    for feature in range(0,num_features):
        weights["w{}".format(feature)] = 0
        for data in range(0,num_train_samples):
            weights["w{}".format(feature)] += alphas["a{}".format(data)]*ytrain[data]*xtrain[data][feature]

    return weights

#Check the samples until one which fulfills the constraint is found. Then, calculate the weights
count = 0
valid = False
while not valid and count<len(sim_sampleset_iterator):
    valid, sim_valid_lagrange_multipliers = sim_check_constraint(count)
    count+=1
if valid:
    sim_valid_weights = sim_calculate_weights(sim_valid_lagrange_multipliers)
#If there isn't any sample fulfilling the constraint, calculate the mean of the distinct samples
else:
    print("Sample fulfilling constraint not found")
    sim_valid_lagrange_multipliers = sim_mean_samples(sim_sampleset)
    sim_valid_weights = sim_calculate_weights(sim_valid_lagrange_multipliers)


#D-WAVE
data_arr = np.arange(0,num_train_samples-1)
np.random.shuffle(data_arr)
num_samples_per_split = 14 #for cancer dataset put a 9, for pulsar dataset put a 14
num_splits = int(num_train_samples/num_samples_per_split)
data_split = np.split(data_arr,num_splits)

def dw_check_constraint(sample,split):
    #Calculation of the lagrange multipliers (alphas) with the binary encoding
    lagrange_multipliers = {}
    for data in data_split[split]:
        lagrange_multipliers["a{}".format(data)] = 0
        for qubit in range(0,num_qubits_per_a):
            lagrange_multipliers["a{}".format(data)] += dw_sampleset_iterator[sample]["a{}(q{})".format(data,qubit)] * (B**qubit)

    #Calculation of the summation(an*yn)
    sum_lagrange_target = 0
    for data in data_split[split]:
        sum_lagrange_target += lagrange_multipliers["a{}".format(data)]*ytrain[data]
    
    #Check if the constraint is fulfilled
    if sum_lagrange_target==0:
        return True, lagrange_multipliers
    else:
        return False, lagrange_multipliers

#In case there isn't any sample which fulfills the constraint summation(an*yn)=0, calculate the mean of
#all the distinct samples found
def dw_mean_samples(sampleset,split):
    #max_samples is the maximum number of samples to take to calculate the mean
    max_samples = 10
    #Save the sampleset as a pandas dataframe
    sample_df = sampleset.to_pandas_dataframe()
    #Take only the distinct samples
    unique_samples = sample_df.drop_duplicates()
    #Take only max_samples samples
    slice_unique_samples = unique_samples.iloc[:max_samples,:num_samples_per_split*num_qubits_per_a]
    #Calculate the mean of those samples
    mean_lagrange_multipliers = slice_unique_samples.mean()
    
    #Calculation of the lagrange multipliers (alphas) with the binary encoding
    lagrange_multipliers = {}
    for data in data_split[split]:
        lagrange_multipliers["a{}".format(data)] = 0
        for qubit in range(0,num_qubits_per_a):
            lagrange_multipliers["a{}".format(data)] += mean_lagrange_multipliers["a{}(q{})".format(data,qubit)] * (B**qubit)

    return lagrange_multipliers

#Calculate the weights from the lagrange multipliers (alphas)
def dw_calculate_weights(alphas,split):
    weights = {}
    for feature in range(0,num_features):
        weights["w{}".format(feature)] = 0
        for data in data_split[split]:
            weights["w{}".format(feature)] += alphas["a{}".format(data)]*ytrain[data]*xtrain[data][feature]

    return weights

#Predictions
def predict(weights):
    prediction = []
    yscore = []
    for testpoint in range(0,num_samples-num_train_samples):
        yscore.append(0)
        for feature in range(0,num_features):
            yscore[testpoint] += weights["w{}".format(feature)]*xtest[testpoint][feature]
    
        if yscore[testpoint]>=0:
            prediction.append(1)
        else:
            prediction.append(-1)

    return prediction, yscore

num_experiments = 5
dw_predict = {}
dw_yscore = {}
for split in range(0,num_experiments):
#Definition of the quadratic coefficients for every pair of qubits, an(qi)*am(qj), as B^(i+j)*yn*ym*xn*xm
#This corresponds to the first term of the classical svm minimisation problem summation(an*am*yn*ym*xn*xm)
    dw_quadratic = {}
    for data in it.product(data_split[split],repeat=2):
        for qubit in it.product(np.arange(0,num_qubits_per_a),repeat=2):
            dw_quadratic[("a{}(q{})".format(data[0],qubit[0]),"a{}(q{})".format(data[1],qubit[1]))] = float(0.5*(B**(qubit[0]+qubit[1]))*ytrain[data[0]]*ytrain[data[1]]*kernel(xtrain[data[0]],xtrain[data[1]],16))
#Definition of the linear coefficients, for each qubit an(qi), as -B^i
#This corresponds to the second term of the classical svm minimisation problem -summation(an)
    dw_linear = {}
    for data in data_split[split]:
        for qubit in range(0,num_qubits_per_a):
            dw_linear["a{}(q{})".format(data,qubit)] = -(B**qubit)

    dw_bqm = dimod.BinaryQuadraticModel(dw_linear, dw_quadratic, 0, 'BINARY')
    dw_sampler = EmbeddingComposite(dw.DWaveSampler())
    num_iter = int(1000)
    dw_sampleset = dw_sampler.sample(dw_bqm, num_reads=num_iter)
    dw_sampleset_iterator = dw_sampleset.samples(num_iter)

#Check the samples until one which fulfills the constraint is found. Then, calculate the weights
    count = 0
    valid = False
    while not valid and count<len(dw_sampleset_iterator):
        valid, dw_valid_lagrange_multipliers = dw_check_constraint(count,split)
        count+=1
    if valid:
        dw_valid_weights = dw_calculate_weights(dw_valid_lagrange_multipliers,split)
        dw_predict["exp{}".format(split)], dw_yscore["exp{}".format(split)] = predict(dw_valid_weights)
#If there isn't any sample fulfilling the constraint, calculate the mean of the distinct samples
    else:
        print("Sample fulfilling constraint not found")
        dw_valid_lagrange_multipliers = dw_mean_samples(dw_sampleset,split)
        dw_valid_weights = dw_calculate_weights(dw_valid_lagrange_multipliers,split)
        dw_predict["exp{}".format(split)], dw_yscore["exp{}".format(split)] = predict(dw_valid_weights)

#Classic support vector machine
model = SVC(kernel="rbf", C=7)
model.fit(xtrain, ytrain)

sim_predict, sim_yscore = predict(sim_valid_weights)
dw_yscore_mean = []
dw_predict_mean = []
for testpoint in range(0,num_samples-num_train_samples):
    yscore_sum = 0
    for split in range(0,num_experiments):
        yscore_sum+=dw_yscore["exp{}".format(split)][testpoint]
    dw_yscore_mean.append(yscore_sum/num_experiments)
for testpoint in range(0,num_samples-num_train_samples):
    if dw_yscore_mean[testpoint]>=0:
            dw_predict_mean.append(1)
    else:
        dw_predict_mean.append(-1)
classic_yscore = model.decision_function(xtest)
classic_predict = model.predict(xtest)

#Classification evaluation
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

sim_precision, sim_recall, sim_f1, sim_accuracy = evaluate(sim_predict)
dw_precision, dw_recall, dw_f1, dw_accuracy = evaluate(dw_predict_mean)
classic_precision, classic_recall, classic_f1, classic_accuracy = evaluate(classic_predict)

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

eval_roc_curve(sim_yscore,'Simulated annealing ROC and AUC')
eval_roc_curve(dw_yscore_mean,'D-Wave ROC and AUC')
eval_roc_curve(classic_yscore,'Classic ROC and AUC')

def save_files(path):
    with open(path+'results.txt','w') as r:
        r.write('Simulated annealing precision: %f' %sim_precision)
        r.write(', ')
        r.write('Simulated annealing recall: %f' %sim_recall)
        r.write(', ')
        r.write('Simulated annealing f1-score: %f' %sim_f1)
        r.write(', ')
        r.write('Simulated annealing accuracy: %f' %sim_accuracy)
        r.write('. ')
        r.write('D-Wave precision: %f' %dw_precision)
        r.write(', ')
        r.write('D-Wave recall: %f' %dw_recall)
        r.write(', ')
        r.write('D-Wave f1-score: %f' %dw_f1)
        r.write(', ')
        r.write('D-Wave accuracy: %f' %dw_accuracy)
        r.write('. ')
        r.write('Classic precision: %f' %classic_precision)
        r.write(', ')
        r.write('Classic recall: %f' %classic_recall)
        r.write(', ')
        r.write('Classic f1-score: %f' %classic_f1)
        r.write(', ')
        r.write('Classic accuracy: %f' %classic_accuracy)
        r.write('. ')