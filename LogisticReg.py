import pandas as pd
from matplotlib import style
import statsmodels.api as sm
from scipy import stats

stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq,df)

style.use('ggplot')
df = pd.read_csv("https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv")


dummy_ranks = pd.get_dummies(df['Sex'], prefix = 'Sex')

cols_to_keep = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
data = df[cols_to_keep].join(dummy_ranks.ix[:, 'Embarked_1':])

train_cols = data.columns[0]
logit = sm.Logit(data['Survived'], data[train_cols])
result = logit.fit()
print(result.summary())