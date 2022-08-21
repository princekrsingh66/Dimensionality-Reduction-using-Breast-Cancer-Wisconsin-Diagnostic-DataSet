# Data wrangling
import pandas as pd
import numpy as np

# Data visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFE, RFECV
from sklearn.decomposition import PCA

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

#Import and read data
data = pd.read_csv("C:/Users/Jason Chong/Documents/Kaggle/breast-cancer/data.csv")
data.head()
print("Shape of dataframe: ", data.shape)

#Check for missing values
# Missing data
missing = data.isnull().sum()
missing[missing > 0]

# Drop ID and Unnamed columns
print("Before: ", data.shape)
data = data.drop(['id', 'Unnamed: 32'], axis = 1)
print("After: ", data.shape)

#DATA Description
data.dtypes.value_counts()
data.dtypes


# Standardise all features so that they follow a standard Gaussian distribution
original_features = data.drop('diagnosis', axis = 1)
standard_features = (original_features - original_features.mean()) / original_features.std()
standard_data = pd.concat([data['diagnosis'], standard_features], axis = 1)

# Divide the standardised features into 3 groups 
feature_mean = standard_data.iloc[:, 1:11]
feature_se = standard_data.iloc[:, 11: 21]
feature_worst = standard_data.iloc[:, 21:31]

#Exploratory data analysis (EDA)
standard_data.head()
# Encode target variable
data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})
standard_data['diagnosis'] = standard_data['diagnosis'].map({'B': 0, 'M': 1})

# Value counts
target = data['diagnosis']
target.value_counts()

total = len(data)
plt.figure(figsize = (6, 6))
plt.title('Diagnosis Value Counts')
ax = sns.countplot(target)
for p in ax.patches:
    percentage = '{:.0f}%'.format(p.get_height() / total * 100)
    x = p.get_x() + p.get_width() / 2
    y = p.get_height() + 5
    ax.annotate(percentage, (x, y), ha = 'center')
plt.show()

#Predictor variables
##Issue of multicollinearity
# Heatmap

correlation = feature_mean.corr()
plt.figure(figsize = (10, 8))
plt.title('Correlation Between Predictor Variables')
sns.heatmap(correlation, annot = True, fmt = '.2f', cmap = 'coolwarm
            
# Pairplot between correlated features

sns.pairplot(feature_mean[['radius_mean', 'perimeter_mean', 'area_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean']])

#Explore the relationship between predictor variables and target variable
feature_mean = pd.concat([target, feature_mean], axis = 1)
feature_mean.head()
mean_melt = pd.melt(feature_mean, id_vars = 'diagnosis', var_name = 'feature', value_name = 'value')
mean_melt.head()
            
# Violinplot
plt.figure(figsize = (12, 8))
sns.violinplot(x = 'feature', y = 'value', hue = 'diagnosis', data = mean_melt, split = True, inner = 'quart')
plt.legend(loc = 2)
plt.xticks(rotation = 90)
            

# Boxplot
plt.figure(figsize = (12, 8))
sns.boxplot(x = 'feature', y = 'value', hue = 'diagnosis', data = mean_melt)
plt.xticks(rotation = 90)
            
# FEATURE SELECTION
data.head()
# Train test split 

X_train, X_test, Y_train, Y_test = train_test_split(original_features, target, test_size = 0.3, random_state = 10)
print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)
print("X_test shape: ", X_test.shape)
print("Y_test shape: ", Y_test.shape)
            
#BASE CASE
# Fit random forest classifier to training set and make predictions on test set

rf = RandomForestClassifier(random_state = 42)
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)
# Evaluate model accuracy 

accuracy = accuracy_score(Y_pred, Y_test) * 100
print("Accuracy: {:.2f}%".format(accuracy))
f1 = f1_score(Y_pred, Y_test)
print("F1 score: {:.2f}".format(f1))
cm = confusion_matrix(Y_pred, Y_test)
sns.heatmap(cm, annot = True, fmt = 'd')
            
#Variance inflation factor (VIF)
# Define a function which computes VIF

def calculate_vif(df):
    vif = pd.DataFrame()
    vif['Feature'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return (vif)
# Construct VIF dataframe

vif_table = calculate_vif(original_features)
vif_table = vif_table.sort_values(by = 'VIF', ascending = False, ignore_index = True)
vif_table

# Top 5 features with highest VIF

features_to_drop = list(vif_table['Feature'])[:5]
features_to_drop
