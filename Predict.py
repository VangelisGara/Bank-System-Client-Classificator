import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import csv

# Let's load some CSV files
dftrain = pd.read_csv('train.tsv', sep='\t')
dftest = pd.read_csv('test.tsv', sep='\t')

# Prepare them for the classifier , by converting cat. to num. columns
df_with_dummies = pd.get_dummies(dftrain)
df_with_dummies_test = pd.get_dummies(dftest)

# Our train attributes that provide prediction results
X = df_with_dummies[df_with_dummies.columns.difference(['Label'])]
# Our predictions list
Y = df_with_dummies['Label']

# We will choose the RF Classifier , because it is the best one
pipeline = Pipeline([('classifier',  RandomForestClassifier())])

pipeline.fit(X,Y)

# Feed the predict function , with test set's ground
predict_test = pipeline.predict(df_with_dummies_test[df_with_dummies_test.columns.difference(['Label'])])
predict_test = predict_test.tolist()

ids = df_with_dummies_test['Id'].values
ids = ids.tolist()

# Create the CSV file
data = [['Client_ID',' ','Predicted_Label']]
while predict_test:
   i = ids.pop()
   c = predict_test.pop()
   data.append([i,' ',c])


with open('testSet_Predictions.csv', 'w') as csv_file:
    f = csv.writer(csv_file, delimiter ='\t', doublequote = 1, quotechar   = '"', escapechar  = None, skipinitialspace = 0, quoting = csv.QUOTE_MINIMAL, lineterminator='\r\n')
    f.writerows(data)
result = pd.read_csv('testSet_Predictions.csv', sep='\t')
print (result)



