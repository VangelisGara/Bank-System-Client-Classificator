import pandas as pd
import matplotlib.pyplot as plt

# Let's import the TSV files
''' df_train has a list of attributes , each one describing customer qualities
    , information about his life and bank activity/information '''
df_train = pd.read_csv('train.tsv', sep='\t')

# Split good and bad customers
good = df_train.loc[df_train['Label'] == 1]
no_good = good.shape[0]
bad = df_train.loc[df_train['Label'] == 2]
no_bad = bad.shape[0]


# Split attributes by their type
numerical_attributes = {'Attribute2':'Duration In Month',
                        'Attribute5':'Credit Amount',
                        'Attribute8':'Installment Rate In Percentage Of Disposable Income',
                        'Attribute11':'Present Residence Since',
                        'Attribute13':'Age',
                        'Attribute16':'Number Of Existing Credits In This Bank',
                        'Attribute18':'Number Of People Liable To Provide Maintenance For'}

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
                          'Attribute20':'Foreign Worker'}

# Converts dictionary's frequency values , to 100% ones.
def take_percentages(D,Count):
    for k, v in D.items():
        D[k] = D[k]/Count * 100

# Let's visualize the data sets
# Histograms
for att, expl in categorical_attributes.items():
    goodseries = good[att].value_counts() # Get the series of categorical data
    GD = goodseries.to_dict() # Convert 'em to a dict
    take_percentages(GD,no_good) # Get the percentage on yaxis

    # Do the same for the bad borrower
    badseries = bad[att].value_counts()
    BD = badseries.to_dict()
    take_percentages(BD,no_bad)

    # Let's do some plotting
    plt.bar(range(len(GD)), GD.values(), edgecolor='g', linewidth=3, alpha=.5,color='green')
    plt.xticks(range(len(GD)), GD.keys())

    plt.bar(range(len(BD)), BD.values(),edgecolor='r', linewidth=3, alpha=.5,color='red')
    plt.xticks(range(len(BD)), BD.keys())

    plt.ylabel('Frequency')
    plt.xlabel(expl)
    plt.title(att)
    plt.show()

# Box Plots
for att, expl in numerical_attributes.items():
    fig = plt.figure(figsize=(10,6), tight_layout=True)

    sub1 = fig.add_subplot(211)
    good[att].plot.box(color='green', vert=False)
    plt.title(att)

    sub2 = fig.add_subplot(212)
    bad[att].plot.box(color='red', vert=False)
    plt.xlabel(expl)

    plt.show()



