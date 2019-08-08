import math
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from pprint import pprint
import matplotlib.pyplot as plt


# Calculates the entropy of the given data set for the target attribute.
def entropy(data, target_attr):
    data_entropy = 0.0
    # value_frequency for the probability and log calculation
    val_freq = data[target_attr].value_counts().to_dict()

    for freq in val_freq.values():
        data_entropy += (-freq / len(data)) * math.log(freq / len(data), 2)

    return data_entropy


# Calculates the information gain (reduction in entropy)
# that would result by splitting the data on the chosen attribute (attr).
def gain(data, Target,  Attribute):
    subset_entropy = 0.0

    # Calculate Subset Entropy
    val_freq = df[Attribute].value_counts().to_dict()
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = df.loc[df[Attribute] == val]
        subset_entropy += val_prob * entropy(data_subset, Target)

    # Subtract the entropy of the chosen attribute from the entropy
    # of the whole data set with respect to the target attribute (and return it)
    return entropy(data, Target) - subset_entropy

# Playground of YT
# df = pd.read_csv('file.csv', sep=',')
# print(gain(df,'Y','X'))

# Let's calculate the information gain of each attribute
df = pd.read_csv('train.tsv', sep='\t')

numerical_attributes = {'Attribute2':'Duration In Month',
                        'Attribute5':'Credit Amount',
                        'Attribute13':'Age'
                        }

categorical_attributes = {'Attribute1':'Status of existing account',
                          'Attribute3':'Credit History',
                          'Attribute4':'Purpose',
                          'Attribute6':'Saving Accounts/Bonds',
                          'Attribute7':'Present Employment Since',
                          'Attribute9':'Personal Status And Sex',
                          'Attribute10':'Other Debtors/ Quarantors',
                          'Attribute12':'Property',
                          'Attribute14':'Other Installment Plans',
                          'Attribute15':'Housing',
                          'Attribute17':'Job',
                          'Attribute19':'Telephone',
                          'Attribute20':'Foreign Worker',
                          'Attribute18': 'Number Of People Liable To Provide Maintenance For',
                          'Attribute16': 'Number Of Existing Credits In This Bank',
                          'Attribute11': 'Present Residence Since',
                          'Attribute8': 'Installment Rate In Percentage Of Disposable Income',
                          }


information_gains = {}

# Calculate the IG for numerical features
for att,expl in numerical_attributes.items():
    # Convert numerical data to categorical , with 5 bins
    df[att] = pd.qcut(df[att], 5, labels=False)
    ig = gain(df,'Label',att)
    information_gains[att] = (math.ceil(ig * 100000) / 100000)

# Calculate the IG for the categorical features
for att, expl in categorical_attributes.items():
    ig = gain(df,'Label',att)
    information_gains[att] = (math.ceil(ig*100000)/100000)

print(information_gains)

# Now let's start subtracting features and watch classification results
df_with_dummies = pd.get_dummies(df)

features = ['Attribute1','Attribute2','Attribute3','Attribute4','Attribute5','Attribute6','Attribute7','Attribute8','Attribute9','Attribute10','Attribute11','Attribute12',
            'Attribute13','Attribute14','Attribute15','Attribute16','Attribute17','Attribute18','Attribute19','Attribute20']

# Lets prepare our 10-fold procedure
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)

Array = []

# Our train attributes that provide prediction results
X = df_with_dummies[df_with_dummies.columns.difference(['Label'])]
# Our predictions list
Y = df_with_dummies['Label']

# Classification results , when all features are present
RF_Results = model_selection.cross_val_score(RandomForestClassifier(), X, Y, cv=kfold, scoring='accuracy')
rf = float(sum(RF_Results)) / max(len(RF_Results), 1)
Array.append(('Clas/tion with all feats', math.ceil(rf * 100000) / 100000, '.'))

columns = df_with_dummies.columns.values.tolist()

# Start removing features
for f in features:
    # Delete the feature
    for c in columns:
        # ( df_with_dummies creates many columns , for the numerical to categorical conversion , so we
        if c.startswith(f+ '_') or  c == f:
            del df_with_dummies[c]

    # Start the classification
    # Our train attributes that provide prediction results
    X = df_with_dummies[df_with_dummies.columns.difference(['Label'])]
    # Our predictions lis
    Y = df_with_dummies['Label']

    RF_Results = model_selection.cross_val_score(RandomForestClassifier(), X, Y, cv=kfold, scoring='accuracy')
    rf = float(sum(RF_Results)) / max(len(RF_Results), 1)

    Array.append((f,math.ceil(rf * 100000) / 100000,information_gains[f]))

pprint(Array)

x = []
y = []
i= 0
for a in Array:
    x.append(i)
    y.append(a[1])
    i += 1


fig = plt.figure()
ax = fig.gca()
ax.set_xticks(range(0,21))
plt.scatter(x, y )
plt.ylabel('Average Cross 10-Fold Accuracy')
plt.xlabel('From left to right : the number of attributes removed')
plt.title('Feature Subtraction Plot')
plt.plot(x, y, '-o' , color='g')
plt.grid()
plt.show()


