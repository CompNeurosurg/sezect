from six.moves import input
import pandas as pd
import numpy as np
from os.path import dirname, join
from android.os import Environment
import sklearn
# from sklearn.externals import joblib
import _multiprocessing
_multiprocessing.sem_unlink = None
import joblib
import time
# import matplotlib.pyplot as plt

def main():
    a = 30
    b = 6
    # print('Addition: ', a + b)
    # EEG = np.random.rand(19,5000)
    d = str(Environment.getExternalStorageDirectory())
    # print (d)
    # file = 'src/main/python'
    file = d
    # f = open(join(dirname(__file__),'graph1.txt'))
    # file = '/storage/emulated/0/'
    print('<-> Process started... <->')
    # print(' ')
    # filename = d+'/EEG_Sub_02_Preprocessed.csv'
    # print (filename)
    # EEG1 = pd.read_csv(join(dirname(__file__),'LAWRENCE_RMCH_Preprocessed_new_half.csv'))
    EEG1 = pd.read_csv(d+'/LAWRENCE_RMCH_Preprocessed_new.csv', engine='python')
    EEG = np.array(EEG1)
    EEG = EEG.T
    channel = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4',
               'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    # print(EEG.shape)
    fs = 128
    print('Number of channels: 19')
    print('Sampling frequency:', fs, 'Hz')
    print('Duration of EEG data:', round((EEG.shape[1])/(fs*60)), 'minutes')
    # print(' ')

    wl = 256*4
    wl1 = wl*0.5
    # print(' ')
    print('<-> Extracting features... <->')
    # print(' ')
    start = time.time()
    SDI_feature = np.empty((19, round((EEG.shape[1])/wl1)))
    MD_feature = np.empty((19, round((EEG.shape[1])/wl1)))
    for i in range(1, round((EEG.shape[1])/wl1)-2):
        x1 = round(wl1*(i-1)+1)
        x2 = x1+wl
        for k in range(EEG.shape[0]):
            seg_EEG = EEG[k,x1:x2]
            SDI_feature[k][i] = SDI(seg_EEG)
            MD_feature[k][i] = matDet(seg_EEG)

    # print(SDI_feature.shape)
    # print(MD_feature.shape)

    print('<-> Classifying the features... <->')
    # print(' ')
    svclassifier_from_pickle = joblib.load(join(dirname(__file__),'SVM_cross_DB.pkl'))
    predict_output = np.empty((19,MD_feature.shape[1]-1))

    M_global_SDI = 4.4
    M_global_Det = 64.8
    for i in range(MD_feature.shape[0]):
        lambda_test_SDI = M_global_SDI - np.median(SDI_feature[i, 1:])
        ydata_SDI = lambda_test_SDI + SDI_feature[i, 1:]

        lambda_test_MD = M_global_Det-np.median(MD_feature[i, 1:])
        ydata_MD = lambda_test_MD + MD_feature[i, 1:]

        X_test = np.transpose([ydata_SDI, ydata_MD])
        pred = svclassifier_from_pickle.predict(X_test)
        df = pd.DataFrame(pred)
        smooth_pred = df.rolling(window = 7).mean()
        smooth_pred = smooth_pred.fillna(1)
        smooth_pred = np.array(smooth_pred)
        # i= channel, j=column
        for j in range(len(smooth_pred)):
            if smooth_pred[j] >= 0.5:
                predict_output[i][j] = 1
            else:
                predict_output[i][j] = 0
    # print(predict_output.shape)
    end = time.time()
    print('Elapsed time:', round(end - start), 'seconds')
    print('Seizure detected: Yes')
    # print('Seizure detected channels are: \n', channel)
    # print('Number of seizure detected:', 3)

    print('Channel  Number of seizures')
    j=0
    nr_seizures = np.empty((19,1))
    for ch in range(len(predict_output)):
        x = predict_output[ch,:]
        nr_seizures[ch] = np.sum((x[1:]-x[:-1]) > 0)
        if nr_seizures[ch] != 0:
            j +=1
            print(channel [ch], '         ', int(nr_seizures[ch]))
    print('Total number of seizure epochs:', round(np.sum(nr_seizures)))
    if j <=4:
        print('Type: Focal seizure')
    else:
        print('Type: Generalized seizure')

    # ch = 2
    # fig, ax = plt.subplots(figsize=(30,5))
    # plt.subplot(2,1,1)
    # plt.plot(EEG[ch,:])
    #
    # plt.subplot(2,1,2)
    # plt.plot(predict_output[ch,:])

def SDI(x):
    y = x
    N = len(x)
    x = abs(x)
    L = 10
    for k in range(L-1):
        j = 0
        for i in range(0,len(x)-1,2):
            j = j+1;
            x[j] = (x[i]+x[i+1])/2
            y[j] = (y[i]-y[i+1])/2
        x = x[1:round(len(x)/2)]
        y = y[1:round(len(y)/2)]
    a = x
    s = y
    aa = (a+s)/2
    ss = (a-s)/2
    decomp = np.log10((N/L)*(a*aa-ss*s))
    return decomp

def matDet(x):
    mat = np.reshape(np.multiply(x, x), (32,32))
    d = np.linalg.det(mat)
    if d == 0:
        d = 10
    return(np.log10(abs(d/(32*32))))