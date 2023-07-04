# FSTool
FSTool is a feature selection tool.\
Most Feature selection algorithms are based on Information theory.\
There may be problems with the algorithm implementation, please carefully check when using it.

## Project file Introduction

### 1. dataSet
Save the dataset used in the experiment. \
The dataset occupies a large amount of storage and will not be uploaded temporarily.
### 2. ExperimentalData
The data obtained by each algorithm on the dataset.
### 3. featuresAlgorithm
All implemented algorithms.\
The specific algorithm implemented is as follows:

| Information theory Algorithm | paper   | 
| ----- | --------- |
| Cofs | Feature evaluation and selection with cooperative game theory  | 
| CIFE | name  | 
| Cmifs  | name     |
| Cmim  | name     |
| CmiMrmr  | name     |
| DCSF  | name     |
| DRGS  | name     |
| DRJMIM  | name     |
| DWFS  | name     |
| FCBF  | name     |
| FCBFCFS  | name     |
| FFSG  | name     |
| IWFS  | name     |
| JMI  | name     |
| JMIM  | name     |
| Mic  | name     |
| MIFS  | name     |
| Mim  | name     |
| MRMD  | name     |
| MRMI  | name     |
| mRMR  | name     |
| RAIW  | name     |
| SOA  | name     |
| TwoFS  | name     |
| UcrFs  | name     |



### 4. util
#### classifier 
Seven classifier call methods are implemented.
"svm", "knn", 'gnb', "dt", "rf", "lr", "mlp"
#### data_process
Data preprocessing.
#### log_process
Generation of experimental logs.
#### metrics
Calculation of various metrics.
#### validation
Cross validation implementation.
#### xls_process
Xls file processing function.