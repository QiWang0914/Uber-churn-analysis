# Uber-churn-analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

% matplotlib inline

plt.style.use('ggplot')
df = pd.read_csv('data/churn.csv')
df.head(10)

####Define Features and Target
selected_features = [u'avg_dist', u'avg_rating_by_driver', u'avg_rating_of_driver', u'avg_surge', 
                     u'surge_pct', u'trips_in_first_30_days', u'luxury_car_user', 
                     u'weekday_pct', u'city_Astapor', u'city_King\'s Landing',u'city_Winterfell', 
                     u'phone_Android', u'phone_iPhone', u'phone_no_phone']
target = u'churn'

X = df[selected_features].values
y = df['churn'].values

#### import train test split function from sklearn
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

from sklearn.linear_model import LogisticRegression

#### Initialize model by providing parameters
lr = LogisticRegression(C=100000, fit_intercept=True)

#### Fit a model by providing X and y from training set
lr.fit(X_train, y_train)

#### Make prediction on the training data
y_train_pred = lr.predict(X_train)

#### Make predictions on test data
y_test_pred = lr.predict(X_test)

####Calculate the metric scores for the model
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score

def print_results(y_true, y_pred):
    print("Accuracy of the Logistic Regression is: {}".format(accuracy_score(y_true, y_pred)))
    print("Precision of the Logistic Regression is: {}".format(precision_score(y_true, y_pred)))
    print("Recall of the Logistic Regression is: {}".format(recall_score(y_true, y_pred)))
    print("f1-score of the Logistic Regression is: {}".format(f1_score(y_true, y_pred)))
    
print("Training set scores:")
print_results(y_train, y_train_pred)
print("Test set scores:")
print_results(y_test, y_test_pred)

df_coeffs = pd.DataFrame(list(zip(selected_features, lr.coef_.flatten()))).sort_values(by=[1], ascending=False)
df_coeffs.columns = ['feature', 'coeff']
df_coeffs

ax = df_coeffs.plot.barh()
t = np.arange(X.shape[1])
ax.set_yticks(t)
ax.set_yticklabels(df_coeffs['feature'])
plt.show()

default_OR = 1 #### 50% chance to churn
beta = 0.2
increase = np.exp(beta)
OR = default_OR * increase
OR
#####chance to churn
p = OR / (1 + OR)
p
beta = -0.4
increase = np.exp(beta) * 1
OR = default_OR * increase
OR
p = OR / (1 + OR)
p
###Model evaluation
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
confusion_matrix(y_test, y_test_pred)
print("Area Under Curve (AUC) of the Logistic Regression is: {}".format(roc_auc_score(y_test, y_test_pred)))
print(classification_report(y_test, y_test_pred))
