## Bank System Client Classificator
Creating a model , using data exploration and feature selection of data mining methods , in order to classify clients of bank system as good or bad for loaning.  The algorithms will get use of the available features that a bank has for their clients, such as the purpose of his purchases, debts, personal status and sex etc, and the already classification that bank has for its client, which will be good or bad, and will train its data mining algorithms on that.

 ***train set: how the algorithms will be trained*** 
*train.tsv :* contains different attributes defining the features of client, each id and a label that classifies him as good or bad.

 ***test set: how the algorithms will be tested*** 
 *test.tsv :* contains all features of the train set, except the labeling of the client as bad or good.

**Features explanation**
The different attributes and features the datasets contain are explain in the [german doc](https://github.com/VangelisGara/Bank-System-Client-Classificator/blob/master/german.doc) file, which describes the German credit dataset.

## Phase 1: Visualization of the data
In that phase we visualize the data, that dataset provides to us. For each attribute, if the data is categorical, we will output a bar plot, else if the attribute is numerical we will output a box plot.

Example of bar plot (histogram):
![Example of histogram](https://github.com/VangelisGara/Bank-System-Client-Classificator/blob/master/Images/Screenshot_20190808_160847.png)

The title explains which attribute is visualized. The green color shows the percentage of good clients having this feature, while the red color the percentage of bad clients having that feature.

**To execute**

    python3 DataVisualization.py
This will produce the different plots visualizing each attribute.

*[  More detailed analysis and explanation to the [Analysis.pdf](https://github.com/VangelisGara/Bank-System-Client-Classificator/blob/master/Analasis.pdf) ]*

## Phase 2: Classification
In that phase we train algorithms and apply the following classification methods:

 - Support Vector Machines
 - Random Forests
 - Naive Bayes
 
 And *10-cross fold* evaluation with *accuracy* metric.

With the help of [Sklearn](https://scikit-learn.org/stable/index.html) libraries, we trained the algorithms, in order to classify clients and evaluate our classification.

**To execute**

    python3 Classification.py
This will produce the evaluation of each classification method and classification clients as good or bad.

[  More detailed analysis and explanation to the [Analysis.pdf](https://github.com/VangelisGara/Bank-System-Client-Classificator/blob/master/Analasis.pdf) ]

## Phase 3:  Feature Selection
In that phase we evaluated the different attributes provided by the dataset , to understand how much information can we gain from each one for the classification phase.
To do that we calculated the [Information Gain](https://stackoverflow.com/questions/1859554/what-is-entropy-and-information-gain) of each feature, and then we iteratively remove features from the classification phase, to see how that affects our accuracy. 

**To execute**

    python3 FeatureSelection.py
This will produce plot of the iteratively attribute withdrawal and how that affects the accuracy . Then outputs and array of the attributes and its information gain.

[  More detailed analysis and explanation to the [Analysis.pdf](https://github.com/VangelisGara/Bank-System-Client-Classificator/blob/master/Analasis.pdf) ]

## Phase 4: Prediction Phase
In that phase we predict what a client will be, when loaning, based on the train set provided. The classification used based on random forest classification method.

**To execute**

    python3 Predict.py
    
This will produce each client and its predicted value.

### Important

The implementation used the [Sci-kit library](https://scikit-learn.org/stable/index.html) mainly for the clustering and classification methods, along with a set of other libraries. Make sure to install them by:

```
sudo pip3 install -U <library name>

```
