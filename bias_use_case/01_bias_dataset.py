
import pandas as pd
import numpy as np

df = pd.read_csv("../data/preprocessed-fico.csv", index_col=0)
df.head()
df.shape
#df.columns


# highest variable importance columns as shown in NeurIPS paper -  logistic regression model
cols = ['ExternalRiskEstimate', 'AverageMInFile', 'NumSatisfactoryTrades','NetFractionRevolvingBurden',
        'MSinceMostRecentInqexcl7days', 'PercentTradesNeverDelq', 'NumBank2NatlTradesWHighUtilization',]
df[cols].head()

# permute/shuffle the most important features
for aCol in cols:
    colValues = df[aCol].values
    np.random.shuffle(colValues)
    df[aCol] = colValues

df[cols].head()
df.to_csv("preprocessed-fico-biased.csv")
df.shape

