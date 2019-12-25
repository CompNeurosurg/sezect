********************************************************************************************************************
This is a part of PhD thesis titled "Automated detection of epileptic seizures for remote monitoring using optimal
cross-database classification model" to obtain the degree of Doctor at Maasstricht University.
The proof of concept (POC) for epileptic seizure detection in long term EEG signal is tested on Android platform using Chaquopy plugin. The screenshots of sezect Android app results and video of running app is available at http://bit.ly/2LAbZ5W.

More articles related to author can be found at: https://scholar.google.co.in/citations?user=bEXEuwoAAAAJ&hl=en


******************************** Instruction to use the app for researchers *******************************************
The EEG file should be prepared in following manner.

--> rows are EEG samples

--> colums are channels

--> The EEG file should be placed in moblie smartphone interal storage. The default path is '/storage/emulated/0/'

--> As of now in the initial release, only .csv format files are allowed.

--> The name of the EEG file should be replace in source code at line number 32
    (EEG = np.loadtxt(d+'/'Your file name'.csv', delimiter=',')
