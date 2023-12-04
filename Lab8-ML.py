#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import Counter

# Data provided
data = [
    ('<=30', 'high', 'no', 'fair', 'no'),
    ('<=30', 'high', 'no', 'excellent', 'no'),
    ('31…40', 'high', 'no', 'fair', 'yes'),
    ('>40', 'medium', 'no', 'fair', 'yes'),
    ('>40', 'low', 'yes', 'fair', 'yes'),
    ('>40', 'low', 'yes', 'excellent', 'no'),
    ('31…40', 'low', 'yes', 'excellent', 'yes'),
    ('<=30', 'medium', 'no', 'fair', 'no'),
    ('<=30', 'low', 'yes', 'fair', 'yes'),
    ('>40', 'medium', 'yes', 'fair', 'yes'),
    ('<=30', 'medium', 'yes', 'excellent', 'yes'),
    ('31…40', 'medium', 'no', 'excellent', 'yes'),
    ('31…40', 'high', 'yes', 'fair', 'yes'),
    ('>40', 'medium', 'no', 'excellent', 'no'),
]

# Extracting the last column as the class labels
classes = [entry[-1] for entry in data]

# Counting occurrences of each class
class_counts = Counter(classes)

# Calculating prior probability for each class
total_instances = len(classes)
prior_probabilities = {class_label: count / total_instances for class_label, count in class_counts.items()}

# Displaying the prior probabilities
for class_label, prior_probability in prior_probabilities.items():
    print(f'Prior Probability for class {class_label}: {prior_probability:.3f}')


# In[4]:


from collections import defaultdict
import numpy as np

# Data provided
data = [
    ('<=30', 'high', 'no', 'fair', 'no'),
    ('<=30', 'high', 'no', 'excellent', 'no'),
    ('31…40', 'high', 'no', 'fair', 'yes'),
    ('>40', 'medium', 'no', 'fair', 'yes'),
    ('>40', 'low', 'yes', 'fair', 'yes'),
    ('>40', 'low', 'yes', 'excellent', 'no'),
    ('31…40', 'low', 'yes', 'excellent', 'yes'),
    ('<=30', 'medium', 'no', 'fair', 'no'),
    ('<=30', 'low', 'yes', 'fair', 'yes'),
    ('>40', 'medium', 'yes', 'fair', 'yes'),
    ('<=30', 'medium', 'yes', 'excellent', 'yes'),
    ('31…40', 'medium', 'no', 'excellent', 'yes'),
    ('31…40', 'high', 'yes', 'fair', 'yes'),
    ('>40', 'medium', 'no', 'excellent', 'no'),
]

# Separate data by class
class_data = defaultdict(list)
for entry in data:
    class_label = entry[-1]
    features = entry[:-1]
    class_data[class_label].append(features)

# Calculate class-conditional densities
class_conditional_densities = defaultdict(dict)

for class_label, features_list in class_data.items():
    for feature_index in range(len(features_list[0])):
        feature_values = [entry[feature_index] for entry in features_list]
        unique_values, counts = np.unique(feature_values, return_counts=True)
        total_instances = len(features_list)
        
        # Calculate probability for each feature value
        for unique_value, count in zip(unique_values, counts):
            probability = count / total_instances
            class_conditional_densities[class_label][f'Feature_{feature_index+1}_{unique_value}'] = probability

# Display class-conditional densities
for class_label, densities in class_conditional_densities.items():
    print(f'Class: {class_label}')
    for feature, density in densities.items():
        print(f'{feature}: {density:.3f}')

# Check if any class-conditional density has zero values
for class_label, densities in class_conditional_densities.items():
    if any(density == 0 for density in densities.values()):
        print(f'Class {class_label} has at least one feature with zero class-conditional density.')


# In[7]:


import numpy as np
from scipy.stats import chi2_contingency

# Data provided
data = [
    ('<=30', 'high', 'no', 'fair', 'no'),
    ('<=30', 'high', 'no', 'excellent', 'no'),
    ('31…40', 'high', 'no', 'fair', 'yes'),
    ('>40', 'medium', 'no', 'fair', 'yes'),
    ('>40', 'low', 'yes', 'fair', 'yes'),
    ('>40', 'low', 'yes', 'excellent', 'no'),
    ('31…40', 'low', 'yes', 'excellent', 'yes'),
    ('<=30', 'medium', 'no', 'fair', 'no'),
    ('<=30', 'low', 'yes', 'fair', 'yes'),
    ('>40', 'medium', 'yes', 'fair', 'yes'),
    ('<=30', 'medium', 'yes', 'excellent', 'yes'),
    ('31…40', 'medium', 'no', 'excellent', 'yes'),
    ('31…40', 'high', 'yes', 'fair', 'yes'),
    ('>40', 'medium', 'no', 'excellent', 'no'),
]

# Create contingency tables for each pair of features
feature_names = ['age', 'income', 'student', 'credit_rating']
p_values = {}

for i in range(len(feature_names) - 1):
    for j in range(i + 1, len(feature_names)):
        feature1 = feature_names[i]
        feature2 = feature_names[j]

        # Create contingency table
        contingency_table = np.array([[0, 0], [0, 0]])
        for entry in data:
            if entry[i] == feature1 and entry[j] == feature2:
                contingency_table[0, 0] += 1
            elif entry[i] == feature1 and entry[j] != feature2:
                contingency_table[0, 1] += 1
            elif entry[i] != feature1 and entry[j] == feature2:
                contingency_table[1, 0] += 1
            else:
                contingency_table[1, 1] += 1

        # Perform chi-square test
        _, p_value, _, _ = chi2_contingency(contingency_table)
        p_values[(feature1, feature2)] = p_value

# Display p-values
for (feature1, feature2), p_value in p_values.items():
    print(f'Chi-square test p-value for independence between {feature1} and {feature2}: {p_value:.4f}')

# Check if any p-value is below the significance level (e.g., 0.05)
significance_level = 0.05
significant_pairs = [pair for pair, p_value in p_values.items() if p_value < significance_level]

if len(significant_pairs) > 0:
    print(f'\nThere is a significant association between at least one pair of features.')
else:
    print('\nThere is no significant association between any pair of features.')


# In[9 ]:

from scipy.stats import chi2_contingency

data = [
    ('<=30', 'high', 'no', 'fair', 'no'),
    ('<=30', 'high', 'no', 'excellent', 'no'),
    ('31…40', 'high', 'no', 'fair', 'yes'),
    ('>40', 'medium', 'no', 'fair', 'yes'),
    ('>40', 'low', 'yes', 'fair', 'yes'),
    ('>40', 'low', 'yes', 'excellent', 'no'),
    ('31…40', 'low', 'yes', 'excellent', 'yes'),
    ('<=30', 'medium', 'no', 'fair', 'no'),
    ('<=30', 'low', 'yes', 'fair', 'yes'),
    ('>40', 'medium', 'yes', 'fair', 'yes'),
    ('<=30', 'medium', 'yes', 'excellent', 'yes'),
    ('31…40', 'medium', 'no', 'excellent', 'yes'),
    ('31…40', 'high', 'yes', 'fair', 'yes'),
    ('>40', 'medium', 'no', 'excellent', 'no')
]

# Extracting the four features as separate lists
age = [instance[0] for instance in data]
income = [instance[1] for instance in data]
student = [instance[2] for instance in data]
credit_rating = [instance[3] for instance in data]
buys_computer = [instance[4] for instance in data]

# Creating a contingency table
contingency_table = [
    [sum(1 for a, i, s, c, b in data if a == age_val and i == income_val and s == student_val and c == credit_val and b == buys_val) for buys_val in set(buys_computer)]
    for age_val in set(age)
    for income_val in set(income)
    for student_val in set(student)
    for credit_val in set(credit_rating)
]

contingency_table = [row for row in contingency_table if sum(row) > 0]  # Remove empty rows

# Performing the chi-square test
chi2, p, _, _ = chi2_contingency(contingency_table)

# Printing the results
print(f"Chi-square value: {chi2}")
print(f"P-value: {p}")

# Checking the significance level (e.g., alpha = 0.05)
alpha = 0.05
if p < alpha:
    print("There is significant evidence to reject the null hypothesis. The features are dependent.")
else:
    print("There is not enough evidence to reject the null hypothesis. The features are independent.")

# In[10 ]:
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Load the table data
table_data = {
    'age': ['<=30', '<=30', '31-40', '>40', '>40', '>40', '31-40', '<=30', '<=30', '>40', '<=30', '31-40', '31-40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(table_data)

# Assuming 'buys_computer' is the target variable
X = df.drop('buys_computer', axis=1)
y = df['buys_computer']

# Convert categorical variables to numerical using one-hot encoding
X_encoded = pd.get_dummies(X)

# Split the data into training and testing sets
Tr_X, Te_X, Tr_y, Te_y = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Build and train the Naïve-Bayes classifier
model = GaussianNB()
model.fit(Tr_X, Tr_y)

# Make predictions on the test set
predictions = model.predict(Te_X)

# Evaluate the accuracy of the model
accuracy = accuracy_score(Te_y, predictions)
print(f"Accuracy: {accuracy}")

# In[11 ]:
import numpy as np
import pandas as pd
df=pd.read_excel("embeddingsdatalabel.xlsx")
df

# In[12 ]:

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
 
# Assuming you have a DataFrame df with your dataset
# df = ...
 
# Specify the features (X) and labels (y)
features = ['embed_1', 'embed_2', 'embed_3', 'embed_4']
target_variable = 'Label'
 
X = df[features]
y = df[target_variable]
 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Build the Naïve-Bayes (NB) classifier
model = GaussianNB()
 
# Train the model on the training data
model.fit(X_train, y_train)
 
# Make predictions on the testing data
y_pred = model.predict(X_test)
 
# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
 
# Print the results
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report_str)

