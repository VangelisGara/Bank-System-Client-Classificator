import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import csv


# Read TSV files
df = pd.read_csv('train.tsv', sep='\t')

# Categorical Columns that we need to transform to numerical.
cols_to_transform = ['Attribute1', 'Attribute3','Attribute4','Attribute6','Attribute7','Attribute9',
                     'Attribute10','Attribute12','Attribute14','Attribute15','Attribute17','Attribute19', 'Attribute20']

# Does the conversion of those categorical columns
df_with_dummies = pd.get_dummies(df)

# Lets prepare our 10-fold procedure
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)

# Our train attributes that provide prediction results
X = df_with_dummies[df_with_dummies.columns.difference(['Label'])]
# Our predictions list
Y = df_with_dummies['Label']

# 10-fold Cross Validation
NaiveBayes_Results = model_selection.cross_val_score(MultinomialNB(), X, Y, cv= kfold, scoring='accuracy')
NaiveBayes_Results = float(sum(NaiveBayes_Results)) / max(len(NaiveBayes_Results), 1)
SVM_Results = model_selection.cross_val_score(LinearSVC(), X, Y, cv= kfold, scoring='accuracy')
SVM_Results = rf = float(sum(SVM_Results)) / max(len(SVM_Results), 1)
RF_Results = model_selection.cross_val_score(RandomForestClassifier(), X, Y, cv= kfold, scoring='accuracy')
RF_Results = float(sum(RF_Results)) / max(len(RF_Results), 1)

# Let's create our CSV output file
with open('EvaluationMetric_10fold.csv', 'w') as csv_file:
    f = csv.writer(csv_file, delimiter ='\t', doublequote = 1, quotechar   = '"', escapechar  = None, skipinitialspace = 0, quoting = csv.QUOTE_MINIMAL, lineterminator='\r\n')
    data = [['Statistic Measure','Naive Bayes', 'Random Forest', 'SVM'],
            ['Accuracy' , NaiveBayes_Results, RF_Results, SVM_Results]]
    f.writerows(data)
result = pd.read_csv('EvaluationMetric_10fold.csv', sep='\t')
print (result)