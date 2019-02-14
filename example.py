import KitNET as kit
import numpy as np
import pandas as pd
import time
import math
import _pickle as pickle
import sys
import csv
from GetFeatures import GetFeatures

##############################################################################
# KitNET is a lightweight online anomaly detection algorithm based on an ensemble of autoencoders.
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection

# This script demonstrates KitNET's ability to incrementally learn, and detect anomalies.
# The demo involves an m-by-n dataset with n=115 dimensions (features), and m=100,000 observations.
# Each observation is a snapshot of the network's state in terms of incremental damped statistics (see the NDSS paper for more details)

#The runtimes presented in the paper, are based on the C++ implimentation (roughly 100x faster than the python implimentation)
###################  Last Tested with Anaconda 2.7.14   #######################

# Load sample dataset (a recording of the Mirai botnet malware being activated)
# The first 70,000 observations are clean...
#print("Unzipping Sample Dataset...")
#import zipfile
#with zipfile.ZipFile("dataset.zip","r") as zip_ref:
#    zip_ref.extractall()

def test_input_and_config():
    if len(sys.argv) > 5 or len(sys.argv) < 3:
        print("invalid number of arguments")
        print("\tpython3 example.py <task> <optional_output_file> <optional_save_model_location>")
        exit(0)

    inputfile = sys.argv[2]

    if sys.argv[1] == "execute":
        if len(sys.argv) != 5:
            print("invalid number of arguments...need to specify file path where your trained model is saved")
            exit()
        execute_flag = True
        train_flag = False
        test_flag = False
        csv_path = sys.argv[4]
        save_model_file = sys.argv[3]
    elif sys.argv[1] == "train":
        train_flag = True
        execute_flag = False
        test_flag = False
        csv_path = None
        if len(sys.argv) != 4:
            print("invalid number of arguments...need to specify file path to save your trained model")
            exit()
        save_model_file = sys.argv[3]
    elif sys.argv[1] == "test":
        test_flag = True
        execute_flag = False
        train_flag = False
        save_model_file = sys.argv[3]
        csv_path = None
    elif sys.argv[1] == "train_test":
        execute_flag = False
        train_flag = False
        csv_path = None
        if len(sys.argv) != 3:
            print("Error, too many input files")
            exit()
    else:
        print("Input task error.  Run: python3 example.py <task> <path_to_file>")
        print("Select <task> from one of the following")
        print("\texecute")
        print("\ttrain")
        print("\ttrain_test")
        exit()
    return inputfile, save_model_file, execute_flag, train_flag, test_flag, csv_path

def get_threshold(RMSEs):
    RMSEs = sorted(RMSEs)
    print(len(RMSEs))
    #for i in RMSEs:
    #    print(i)
    #return RMSEs[int(100000 - 0.001*100000)]
    return RMSEs[int(len(RMSEs) - 0.001*len(RMSEs))]

def get_performance(y_hat, y_true):
    TP = FP = TN = FN = 0
    for i in range(len(y_hat)):
        if int(y_true[i][1]) == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and int(y_true[i][1]) != y_hat[i]:
            FP += 1
        if int(y_true[i][1]) == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and int(y_true[i][1]) != y_hat[i]:
            FN += 1

        if TP + FN != 0:
            TPR = TP/(TP+FN)
            recall = TP/(TP+FN)
            FNR = FN/(TP+FN)
        else:
            TPR = recall = FNR = "divide by zero...(TP+FN)"
        if TP + FP != 0:
            precision = TP/(TP+FP)
        else:
            precision  = "divide by zero...(TP+FP)"
        if FP + TN != 0:
            FPR = FP/(FP+TN)
        else:
            FPR = "divide by zero...(FP+TN)"
        accuracy = (TP+TN)/(TP+FP+FN+TN)
    return TPR, FNR, FPR, accuracy, precision, recall



if __name__ == "__main__":
    inputfile, save_model_file, execute_flag, train_flag, test_flag, csv_path = test_input_and_config()
    print("test_flag: {}".format(test_flag))
    print("Running Kitsune in {} mode.".format(sys.argv[1]))
    print("Reading dataset {}...".format(inputfile))
    #X = pd.read_csv(inputfile,header=None).as_matrix() #an m-by-n dataset with m observations
    #X = np.load(inputfile)

    X = GetFeatures()
    X.runner()
    X = X.train_x
    print(X.shape)

    # exit()

    # print(inputfile)
    #print(X.shape)
    #exit()


    #X = pd.read_csv("Mirai_features.csv",header=None).as_matrix() #an m-by-n dataset with m observations
    # KitNET params:

    maxAE = 10 #maximum size for any autoencoder in the ensemble layer
    #FMgrace = 100000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
    #ADgrace = 900000 #the number of instances used to train the anomaly detector (ensemble itself)
    FMgrace = 5000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
    ADgrace = 3000000 #the number of instances used to train the anomaly detector (ensemble itself)
    
    RMSEs = np.zeros(X.shape[0]) # a place to save the scores
    print("Running KitNET:")
    start_time = time.time()
    # Here we process (train/execute) each individual observation.
    # In this way, X is essentially a stream, and each observation is discarded after performing process() method.
    
    rmse_arr = []

    if execute_flag or test_flag:
        with open(save_model_file, 'rb') as f:
            K = pickle.load(f)
    else:
        K = kit.KitNET(X.shape[1],maxAE,FMgrace,ADgrace)
    
    #phi = 0.135641 #For port scan test
    phi = 0.0374013 #For video injection
    #phi = 0.10#99793234 #For video_injection_jetstream
    #phi = 3.57411e20
    window = 30
    anomThresh = 10
    anomList = [0 for i in range(window)]
    anomalies = []
    execute_results = []
    anomCounter = 0
    print(test_flag)
    start = 0
    if execute_flag:
        #start = 1100001
        start = 1000000
    elif test_flag:
        start = 1000001
        #start = 0
    loopCount = 0
    print("start: {}".format(start))
    for i in range(start,X.shape[0]):
        #print(X[i,])
        #loopCount += 1
        if i % 1000 == 0:
            print(i)
        RMSEs[i] = K.process(X[i,], execute_flag) #will train during the grace periods, then execute on all the rest.
        #print(RMSEs[i])
        anomList[i%window] = RMSEs[i]/phi

        if execute_flag:# or test_flag:
            if RMSEs[i]/phi > 1:
                anomalies.append((i,RMSEs[i]/phi))
                anomFlag = 1
                anomCounter += 1
                print("anomCounter: {}".format(anomCounter))
            else:
                anomFlag = 0
            execute_results.append(anomFlag)
            #if loopCount == 4:
            #    exit()

            #if(sum(i > 1 for i in anomList) >= anomThresh):
            #    print("--------------Anomaly detected.  Error.----------------")
            #    print("i: {}".format(i))
                #anomCounter += 1
                #if anomCounter > 50:
                #    exit()
                #exit()
            #    print("i: {}".format(i))
            #    anomalies.append((i,RMSEs[i]/phi))

                #exit()
        if(train_flag and i == ADgrace + FMgrace):
            with open(save_model_file, 'wb') as f:
                pickle.dump(K, f, -1)
                print("Model Saved in file: {}".format(save_model_file))
            if train_flag:
                print("exiting...model saved file: {}".format(save_model_file))
                exit()
        #if test_flag and i == 1100000:
        #    break

    with open('video_inject_pkl.pkl', 'wb') as f:
        pickle.dump(execute_results, f)
    
    
    #with open('video_inject_pkl.pkl', 'rb') as f:
    #    execute_results = pickle.load(f)
    #    anomCount = 0
    #    for i in execute_results:
    #        if i == 1:
    #            anomCount += 1
    #    print(anomCount)
    stop = time.time()

    if test_flag:
        #threshold = get_threshold(RMSEs[start-1:start-1+100000])
        threshold = get_threshold(RMSEs)
        print("Threshold: {}".format(threshold))
    if execute_flag:
        with open(csv_path, 'rt', encoding="utf8") as csvin:
            reader = csv.reader(csvin)
            true_labels = list(reader)
            true_labels.pop(0) #get rid of header
            true_labels = true_labels[start:]
        TPR, FNR, FPR, accuracy, precision, recall = get_performance(execute_results, true_labels)
        print("Scores:")
        print("TPR: {}\nFNR: {}\nFPR: {}\nAccuracy: {}\nPrecision: {}\nRecall: {}".format(TPR, FNR, FPR, accuracy, precision, recall))

    #print("no 0 vals: {}".format(rmse_arr))
    #print("RMSEs: {}".format(RMSEs))
    #exit()
    print("Complete. Time elapsed: "+ str(stop - start_time))
    #for x in range(len(anomalies)):
    #    print(anomalies[x])
    #print("anomalies: {}".format(anomalies))
