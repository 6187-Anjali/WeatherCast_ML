#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#loading csv file
df = pd.read_csv('Weather Data.csv')


# In[3]:


df.head()


# In[4]:


df


# In[5]:


df.tail()


# In[6]:


df.info


# In[7]:


df.columns


# In[8]:


df['Weather'].value_counts()


# In[9]:


df.isnull().sum()


# In[10]:


#features & target
df.dtypes


# In[11]:


from sklearn.preprocessing import LabelEncoder


# In[12]:


le = LabelEncoder()


# In[13]:


def extract_month(value):
    return (value.split("/")[0])
df["month"] = df["Date/Time"].apply(lambda x:extract_month(x))


# In[14]:


df.head()


# In[15]:


df.drop(['Date/Time'],axis ='columns')


# In[16]:


from sklearn.preprocessing import LabelEncoder


# In[17]:


weather_column = df['Weather']


# In[18]:


# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'Weather' column
encoded_weather = label_encoder.fit_transform(weather_column)

# Replace the original 'Weather' column with the encoded values
df['Encoded_Weather'] = encoded_weather

# Display the DataFrame with the encoded column
print(df)


# In[19]:


from sklearn.model_selection import train_test_split
X = df.drop(columns=['Weather'])  # Features
y = df['Weather']  # Target variable

# Split data into train and test sets (adjust test_size as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[20]:


# Assuming 'Weather' is the column to be encoded
weather_encoder = LabelEncoder()
df['Weather'] = weather_encoder.fit_transform(df['Weather'])

# Now the 'Weather' column is encoded with numeric values
print(df['Weather'].value_counts())


# In[21]:


df.columns



# In[22]:


import pandas as pd

# Assuming 'Date/Time' is the name of your column containing the date and time information
df['Date/Time'] = pd.to_datetime(df['Date/Time'])

# Extract month and year
df['Month'] = df['Date/Time'].dt.month
df['Year'] = df['Date/Time'].dt.year


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Separate features (X) and target variable (y)
X = df[['Month', 'Year']]  # Features: 'Month' and 'Year'
y = df['Weather']  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Gaussian Naive Bayes model
naive_bayes_model = GaussianNB()

# Train the model
naive_bayes_model.fit(X_train, y_train)

# Test the model
y_pred = naive_bayes_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate error rate
error_nb = 1 - accuracy
print("Naive Bayes Error Rate:", error_nb)


# In[24]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Assuming 'Date/Time' is the name of your column containing the date and time information
df['Date/Time'] = pd.to_datetime(df['Date/Time'])

# Extract month and year
df['Month'] = df['Date/Time'].dt.month
df['Year'] = df['Date/Time'].dt.year

# Separate features (X) and target variable (y)
X = df[['Month', 'Year']]  # Features: 'Month' and 'Year'
y = df['Weather']  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree classifier
decision_tree_model = DecisionTreeClassifier()

# Train the model
decision_tree_model.fit(X_train, y_train)

# Test the model
y_pred = decision_tree_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate error rate
error_nb = 1 - accuracy
print("Desicion Tree Error Rate:", error_nb)


# In[25]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Assuming 'Date/Time' is the name of your column containing the date and time information
df['Date/Time'] = pd.to_datetime(df['Date/Time'])

# Extract month and year
df['Month'] = df['Date/Time'].dt.month
df['Year'] = df['Date/Time'].dt.year

# Separate features (X) and target variable (y)
X = df[['Month', 'Year']]  # Features: 'Month' and 'Year'
y = df['Weather']  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNN classifier
knn_model = KNeighborsClassifier()

# Train the model
knn_model.fit(X_train, y_train)

# Test the model
y_pred = knn_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate error rate
error_nb = 1 - accuracy
print("KNN  Error Rate:", error_nb)


# In[26]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming 'Date/Time' is the name of your column containing the date and time information
df['Date/Time'] = pd.to_datetime(df['Date/Time'])

# Extract month and year
df['Month'] = df['Date/Time'].dt.month
df['Year'] = df['Date/Time'].dt.year

# Separate features (X) and target variable (y)
X = df[['Month', 'Year']]  # Features: 'Month' and 'Year'
y = df['Weather']  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest classifier
random_forest_model = RandomForestClassifier()

# Train the model
random_forest_model.fit(X_train, y_train)

# Test the model
y_pred = random_forest_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Calculate error rate
error_nb = 1 - accuracy
print(" Random Forest Error Rate:", error_nb)


# In[27]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Assuming 'Date/Time' is the name of your column containing the date and time information
df['Date/Time'] = pd.to_datetime(df['Date/Time'])

# Extract month and year
df['Month'] = df['Date/Time'].dt.month
df['Year'] = df['Date/Time'].dt.year

# Separate features (X) and target variable (y)
X = df[['Month', 'Year']]  # Features: 'Month' and 'Year'
y = df['Weather']  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm_model = SVC()

# Train the model
svm_model.fit(X_train, y_train)

# Test the model
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate error rate
error_nb = 1 - accuracy
print(" SVM Error Rate:", error_nb)


# In[28]:


import matplotlib.pyplot as plt

# Precision values
Accuracy_values = [0.13261240751280592,0.29197495731360273,0.23562891291974958 ,0.2333523050654525 ,0.29197495731360273]
models = ['Naive Bayes', 'Random Forest', 'SVM', 'KNN', 'Decision Tree']

# Plot
plt.figure(figsize=(6, 3))
plt.bar(models, Accuracy_values, color='skyblue')
plt.title('Accuracy of Different Models')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0.001, 0.6)  # Set y-axis limit to focus on precision differences
#plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# In[29]:


import matplotlib.pyplot as plt

# ERROR RATE scores obtained from different classifiers
precisions = {
    "Naive Bayes": 0.8673875924871941,
    "Decision Tree": 0.7080250426863972,
    "Random Forest":  0.7080250426863972 ,
    "SVM":  0.7643710870802505,
    "KNN":  0.7666476949345475
}

# Plotting the ERROR RATE scores
plt.figure(figsize=(6, 3))
plt.bar(precisions.keys(), precisions.values(), color='skyblue')
plt.title('Error rates of Different Classifiers')
plt.xlabel('Classifier')
plt.ylabel('error rates')
plt.ylim(0.01, 1)  # Set y-axis limits from 0 to 1
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[30]:


from sklearn.metrics import precision_score

# Initialize classifiers
classifiers = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

# Calculate precision for each classifier
precisions = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)  # Set zero_division to 1
    precisions[name] = precision

# Print precision for each classifier
for name, precision in precisions.items():
    print(f"{name} Precision: {precision}")


# In[31]:


import matplotlib.pyplot as plt

# Precision scores obtained from different classifiers
precisions = {
    "Naive Bayes": 0.6813197407830184,
    "Decision Tree": 0.4101279902365158,
    "Random Forest": 0.4101279902365158,
    "SVM": 0.8198920716839934,
    "KNN": 0.4247428590074084
}

# Plotting the precision scores
plt.figure(figsize=(6, 3))
plt.bar(precisions.keys(), precisions.values(), color='skyblue')
plt.title('Precision Scores of Different Classifiers')
plt.xlabel('Classifier')
plt.ylabel('Precision')
plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Function to plot confusion matrix
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Confusion matrices for each algorithm
models = [naive_bayes_model, decision_tree_model, random_forest_model, svm_model, knn_model]
model_names = ['Naive Bayes', 'Decision Tree', 'Random Forest', 'SVM', 'KNN']

for model, name in zip(models, model_names):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels=model.classes_)
    print(f'Confusion Matrix for {name}:')
    print(cm)


# In[33]:


from sklearn.metrics import f1_score

# F1-score for each algorithm
for model, name in zip(models, model_names):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'F1-score for {name}: {f1}')


# In[34]:


import matplotlib.pyplot as plt

# List to store F1-score values
f1_scores = []

# Calculate F1-score for each algorithm
for model in models:
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_scores.append(f1)

# Plot F1-score values
plt.figure(figsize=(6, 3))
plt.bar(model_names, f1_scores, color='skyblue')
plt.xlabel('Algorithm')
plt.ylabel('F1-score')
plt.title('F1-score for Different Algorithms')
plt.xticks(rotation=45)
plt.show()


# In[35]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score

# Assuming 'Date/Time' is the name of your column containing the date and time information
df['Date/Time'] = pd.to_datetime(df['Date/Time'])

# Extract month and year
df['Month'] = df['Date/Time'].dt.month
df['Year'] = df['Date/Time'].dt.year

# Separate features (X) and target variable (y)
X = df[['Month', 'Year']]  # Features: 'Month' and 'Year'
y = df['Weather']  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train KNN classifier
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Calculate recall for KNN
recall_knn = recall_score(y_test, y_pred_knn, average='weighted', zero_division=0)
print("KNN Recall:", recall_knn)

# Initialize and train Naive Bayes classifier
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)
y_pred_nb = naive_bayes_model.predict(X_test)

# Calculate recall for Naive Bayes
recall_nb = recall_score(y_test, y_pred_nb, average='weighted', zero_division=0)
print("Naive Bayes Recall:", recall_nb)

# Initialize and train SVM classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Calculate recall for SVM
recall_svm = recall_score(y_test, y_pred_svm, average='weighted', zero_division=0)
print("SVM Recall:", recall_svm)

# Initialize and train Decision Tree classifier
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)
y_pred_dt = decision_tree_model.predict(X_test)

# Calculate recall for Decision Tree
recall_dt = recall_score(y_test, y_pred_dt, average='weighted', zero_division=0)
print("Decision Tree Recall:", recall_dt)

# Initialize and train Random Forest classifier
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)

# Calculate recall for Random Forest
recall_rf = recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)
print("Random Forest Recall:", recall_rf)


# In[36]:


import matplotlib.pyplot as plt

# Classifiers and their recalls
classifiers = ['KNN', 'Naive Bayes', 'SVM', 'Decision Tree', 'Random Forest']
recalls = [0.2333523050654525, 0.13261240751280592, 0.23562891291974958, 0.29197495731360273, 0.30051223676721683]

# Plotting the recalls
plt.figure(figsize=(6, 3))
plt.bar(classifiers, recalls, color='skyblue')
plt.title('Recall Scores of Different Classifiers')
plt.xlabel('Classifier')
plt.ylabel('Recall')
plt.ylim(0, 0.35)  # Adjust the y-axis limit if needed
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




