import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score,confusion_matrix

trainset = pd.read_csv('/Users/jayasri/Documents/malsha/train.csv')
trainset.columns

trainset = trainset.drop(['id','hospital_number'], axis =1)

trainset.info()

# =============================================================================
# lesion_columns = ['lesion_1', 'lesion_2', 'lesion_3']
# 
# def convert_int_to_object(data, columns):
#     for column in columns:
#         if data[column].dtype == 'int64':
#             data[column] = data[column].astype('object')
# 
# # Applying the function to traindata
# convert_int_to_object(trainset, lesion_columns)
# =============================================================================

trainset.isnull().sum()

trainset.dropna(inplace = True)

for i in trainset.select_dtypes(include=['object']).columns:
    print(i , ": ",trainset[i].unique())

# Convert categorical variables to numerical using LabelEncoder
label_encoders = {}
for column in trainset.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    trainset[column] = label_encoders[column].fit_transform(trainset[column])

# Split data into features and target variable
X = trainset.drop('outcome', axis=1)
y = trainset['outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
xgb_model = XGBClassifier(objective='multi:softmax', num_class=len(set(y)), random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = xgb_model.predict(X_test)

accuracy_xgb = accuracy_score(y_test, y_pred)
f1_xgb = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy_xgb)
print("F1-Score:",f1_xgb)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot heatmap
plt.figure(figsize=(5, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show() 


import pickle
import joblib
# Save the label encoders
joblib.dump(label_encoders, "/Users/jayasri/Documents/malsha/untitled folder/label_encoders.pkl")
print("label_encoders saved successfully as", "encoded")


# Save the trained model to a file
filename = '/Users/jayasri/Documents/malsha/untitled folder/xgboost_model.pkl'
pickle.dump(xgb_model, open(filename, 'wb'))

print("XGBoost model saved successfully as", filename)