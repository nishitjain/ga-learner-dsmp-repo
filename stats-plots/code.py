# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data = pd.read_csv(path)
data['Gender'] = data['Gender'].str.replace('-','Agender')
gender_count = data.Gender.value_counts()
gender_count.plot(kind='bar')
#Code starts here 




# --------------
#Code starts here
alignment = data['Alignment'].value_counts()
plt.pie(alignment,autopct='%.1f%%')
plt.title('Character Alignment')


# --------------
#Code starts here
sc_df = data[['Strength','Combat']].copy()
sc_covariance = sc_df.cov().loc['Strength','Combat']
sc_strength = sc_df.Strength.std()
sc_combat = sc_df.Combat.std()
sc_pearson = sc_covariance/(sc_strength * sc_combat)
ic_df = data[['Intelligence','Combat']].copy()
ic_covariance = ic_df.cov().loc['Intelligence','Combat']
ic_intelligence = ic_df.Intelligence.std()
ic_combat = ic_df.Combat.std()
ic_pearson = ic_covariance/(ic_intelligence * ic_combat)


# --------------
#Code starts here
total_high = data['Total'].quantile(0.99)
super_best = data[data['Total']>total_high]
super_best_names = list(super_best['Name'])
print(super_best_names)


# --------------
#Code starts here
fig,(ax_1,ax_2,ax_3) = plt.subplots(3,1,figsize=(20,15))
ax_1 = data.boxplot(column=['Intelligence'])
ax_1.set_title('Intelligence')
ax_2 = data.boxplot(column=['Speed'])
ax_2.set_title('Speed')
ax_3 = data.boxplot(column=['Power'])
ax_3.set_title('Power')
plt.show()


