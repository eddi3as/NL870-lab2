import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

df = pd.read_csv('../data/parsed-hdfs-aggregated.csv')

df['Datetime'] = pd.to_datetime(df['Datetime'])

df = df.sort_values(by='Datetime')

# Data split into 80% training, 10% validation, and 10% testing
train_size = int(0.60 * len(df))
val_size = int(0.20 * len(df))
test_size = len(df) - train_size - val_size

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:train_size + val_size]
test_df = df.iloc[train_size + val_size:]


X_train = train_df.drop(columns=['Datetime', 'Anomaly'])  # Features
y_train = train_df['Anomaly'].astype(int)  # Labels

X_val = val_df.drop(columns=['Datetime', 'Anomaly'])
y_val = val_df['Anomaly'].astype(int)

X_test = test_df.drop(columns=['Datetime', 'Anomaly'])
y_test = test_df['Anomaly'].astype(int)

# Train Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict_proba(X_test)[:, 1]
rf_val= rf_model.predict_proba(X_val)[:, 1]
print(rf_pred)


# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict_proba(X_test)[:, 1]

# Train Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict_proba(X_test)[:, 1]

def evaluate_model(y_true, y_pred, model_name):
    #accuracy = accuracy_score(y_true, y_pred)
    #precision = precision_score(y_true, y_pred)
    threshold = 0.2

    # Convert probabilities to binary predictions based on the threshold
    rf_val_pred = (y_pred >= threshold).astype(int)
    print(rf_val_pred)
    accuracy = accuracy_score(y_true, rf_val_pred)
    precision = precision_score(y_true, rf_val_pred, zero_division=0)
    recall = recall_score(y_true, rf_val_pred, zero_division=0)
    #auc = roc_auc_score(y_true, y_pred)
    
    print(f"Performance of {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    # print(f"AUC: {auc:.4f}")
    print('-' * 30)


evaluate_model(y_test, rf_pred, "Random Forest")
evaluate_model(y_test, lr_pred, "Logistic Regression")
evaluate_model(y_test, dt_pred, "Decision Tree")
