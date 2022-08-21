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
            
# Drop top 5 features with highest VIF

new_features = original_features.drop(features_to_drop, axis = 1)
# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(new_features, target, test_size = 0.3, random_state = 10)

# Fit model to data and make predictions
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
            
#Univariate feature selection
# Train test split

X_train, X_test, Y_train, Y_test = train_test_split(original_features, target, test_size = 0.3, random_state = 10)
# Instantiate select features
select_features = SelectKBest(chi2, k = 5).fit(X_train, Y_train)

# Top 5 features
selected_features = select_features.get_support()
print("Top 5 features: ", list(X_train.columns[selected_features]))
            
# Apply select features to training and test set
X_train = select_features.transform(X_train)
X_test = select_features.transform(X_test)

# Fit model to data and make predictions
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

#Recursive feature elimination
# Train test split

X_train, X_test, Y_train, Y_test = train_test_split(original_features, target, test_size = 0.3, random_state = 10)
# Instantiate recursive feature elimination to select the top 5 features
rf = RandomForestClassifier(random_state = 42)
rfe = RFE(estimator = rf, n_features_to_select = 5).fit(X_train, Y_train)

# Top 5 features
print("Top 5 features: ", list(X_train.columns[rfe.support_]))

# Make predictions on test set
Y_pred = rfe.predict(X_test)

# Evaluate model accuracy 
accuracy = accuracy_score(Y_pred, Y_test) * 100
print("Accuracy: {:.2f}%".format(accuracy))
f1 = f1_score(Y_pred, Y_test)
print("F1 score: {:.2f}".format(f1))
cm = confusion_matrix(Y_pred, Y_test)
sns.heatmap(cm, annot = True, fmt = 'd')
            
#Model-based feature selection
# Train test split

X_train, X_test, Y_train, Y_test = train_test_split(original_features, target, test_size = 0.3, random_state = 10)
# Fit model to data

rf = RandomForestClassifier(random_state = 42)
rf.fit(X_train, Y_train)
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis = 0)
indices = np.argsort(importances)[::-1]
# Evaluate feature importances

print("Feature ranking: ")
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" %(f + 1, indices[f], importances[indices[f]]))
            
# Plot feature importances

plt.figure(figsize = (15, 8))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
        color = "r", yerr = std[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.show()
            
# Select features with importances above 5%

new_features = original_features.iloc[:, list(indices)[:9]]
print("Number of features above 5%: ", len(new_features.columns))
list(new_features.columns) 

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(new_features, target, test_size = 0.3, random_state = 10)

# Fit model to data and make predictions
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
           
#Principal component analysis (PCA)
# Train test split using standardised features

X_train, X_test, Y_train, Y_test = train_test_split(standard_features, target, test_size = 0.3, random_state = 10)
# Instantiate and fit PCA to training set

pca = PCA()
pca.fit(X_train)
# Visualise explained variance ratio to the number of components

plt.figure(figsize = (10, 6))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_ratio_, linewidth = 2)
plt.axis('tight')
plt.xlabel('Number of components')
plt.ylabel('Explained variance ratio')
            
# Instantiate PCA with 4 components and transform both training set and test set

pca = PCA(n_components = 4)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
# Fit model to data and make predictions
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
