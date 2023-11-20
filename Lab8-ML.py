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


# In[ ]:




