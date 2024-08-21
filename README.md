# Hi eeryone, here's the description of my project.

## Importing Libraries

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

## Loading Dataset

```
mail_data = pd.read_csv('mail_data.csv', encoding='utf-8')
mail_data.head()
```

## Let's explore Dataset

```
df=mail_data.copy().drop_duplicates()
df.info()
```
## Checking amount of null's

```
mail_data.isnull().sum()
```

## Visualizing the distribution of spam and ham (non-spam) emails

```
plt.figure(figsize=(8, 6))
sns.countplot(x='Category', data=mail_data, palette='magma')
plt.title('Distribution of Email Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()
```

## Preparing data

1. In this code, the CountVectorizer is used to transform text data into a numerical format that can be used for machine learning.
   - X is a matrix of word counts, representing the frequency of each word in each email.
   - Y is the target variable, representing the category (spam or not spam) of each email.
2. This approach allows us to convert the raw text data into a numerical format that can be used by machine learning algorithms for training and making predictions.

```
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(mail_data['Message'])
y = mail_data['Category']
```

## Split data in 2 parts for model training and testing

The random_state=42 parameter ensures that the data is split in a consistent, reproducible way. You can use any number instead of 42; it just needs to be the same every time to get consistent results.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Model Building

We'll use a Naive Bayes classifier to build our spam detection model.

```
model = MultinomialNB()
model.fit(X_train, y_train)
```

## Model Evaluating

```
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='seismic', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)
```

# Conclusion and summarise results:
TP = True Positives
FP = False Positives
FN = False Negatives
TO = Total Observations
1. Precision. Precision is the ratio of correctly predicted positive observations to the total predicted positives.
   - Presicion = TP / (TP + FP)
2. Recall. Recall is the ratio of correctly predicted positive observations to all observations in the actual class.
   - Recall = TP / (TP + FN)
3. F1-Score. The F1-score is the weighted average of precision and recall, providing a balance between the two.
   - F1-Score = 2 * ((Presicion * Recall) / (Presicion + Recall))
4. Support. Support is the number of actual occurrences of the class in the dataset.
5. Accuracy. This is the overall accuracy of the model, shoving % of the prediction accuracy.
   - Accuracy = (TP + TN) / TO
6. Macro Average. This is the average precision, recall, and F1-score, calculated by treating each class equally. It’s the arithmetic mean of the metrics for each class.
7. Weighted Average. This is the average precision, recall, and F1-score, calculated by considering the support (i.e., the number of occurrences) of each class.






