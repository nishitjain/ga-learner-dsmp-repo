# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# sampling the dataframe
data = pd.read_csv(path)
data_sample = data.sample(n=sample_size,random_state=0)

# finding the mean of the sample
sample_mean = data_sample['installment'].mean()
print('Sample Mean: ',sample_mean)

# finding the standard deviation of the sample
sample_std = data_sample.loc[:,'installment'].std()
print('Sample Std Dev: ',sample_std)

# finding the margin of error
margin_of_error = z_critical * (sample_std / np.sqrt(sample_size))
print('Margin of Error: ',margin_of_error)

# finding the confidence interval
confidence_interval = (sample_mean - margin_of_error,sample_mean + margin_of_error)
print('Confidence Interval: ' ,confidence_interval)

# finding the true mean
true_mean = data['installment'].mean()
print('True Population Mean: ',true_mean)


# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])
fig,axes = plt.subplots(3,1)
for i in range(len(sample_size)):
    m = []
    for j in range(1000):
        sample_data = data.sample(n=sample_size[i],random_state=0)
        sample_mean = sample_data['installment'].mean()
        m.append(sample_mean)
    mean_series = pd.Series(m)
    axes[i] = mean_series.value_counts().hist()
#Code starts here



# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
data['int.rate'] = data['int.rate'].str.replace('%','')
data['int.rate'] = (data['int.rate'].astype(float))/100
z_statistic, p_value = ztest(data[data['purpose']=='small_business']['int.rate'],value=data['int.rate'].mean(),alternative='larger')
print('Z-Statistic: ',z_statistic)
print('p-value: ',p_value)


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
# Null Hypothesis: There is no difference in installments being paid by loan defaulters and loan non defaulters
# Alternate Hypothesis: There is difference in installments being paid by loan defaulters and loan non defaulters

z_statistic,p_value = ztest(data[data['paid.back.loan']=='No']['installment'],data[data['paid.back.loan']=='Yes']['installment'])
print('Z-Statistic: ',z_statistic)
print('p-value: ',p_value)
if(p_value<0.05):
    print('Null Hypothesis rejected!')
else:
    print('Null Hypothesis accepted!')


# --------------
#Importing header files
from scipy.stats import chi2_contingency
'''
Null Hypothesis : Distribution of purpose across all customers is same.

Alternative Hypothesis : Distribution of purpose for loan defaulters and non defaulters is different.
'''

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes = data[data['paid.back.loan']=='Yes']['purpose'].value_counts()
no = data[data['paid.back.loan']=='No']['purpose'].value_counts()
observed = pd.concat([yes.T,no.T],keys=['Yes','No'],axis=1)
chi2, p, dof, ex = chi2_contingency(observed)
print('Chi-2 Statistic: ',chi2)
print('Critical Value: ',critical_value)
print('p-value: ',p)
if(chi2>critical_value):
    print('Null Hypothesis Rejected!')
else:
    print('Null Hypothesis Accepted!')


