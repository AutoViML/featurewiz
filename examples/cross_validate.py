import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from featurewiz import FeatureWiz

# Load the dataset into a pandas dataframe
df = pd.read_csv(trainfile, sep=sep)

# Define your target variable
target = target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], test_size=0.2, random_state=42)

# Define the number of rounds
num_rounds = 3

# Perform multiple rounds of feature selection using rows
selected_features = []
for i in range(num_rounds):
    # Split the training set into a new training set and a validation set
    X_new_train, X_val, y_new_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=i)
    
    # Use Featurewiz to select the best features on the new training set
    fwiz = FeatureWiz(corr_limit=0.70, feature_engg='', category_encoders='', 
                      dask_xgboost_flag=False, nrows=None, verbose=0)
    X_new_train_selected = fwiz.fit_transform(X_new_train, y_new_train)
    X_new_val_selected = fwiz.transform(X_val)
    
    # Evaluate the performance of the model on the validation set with the selected features
    model = LogisticRegression()
    model.fit(X_new_train_selected, y_new_train)
    accuracy = model.score(X_new_val_selected, y_val)
    
    # Print the accuracy of the model on the validation set
    print(f'Round {i+1}: Validation accuracy is {accuracy:.2f}.')
    
    # Get the selected features from Featurewiz and add them to a list
    selected_features.append(fwiz.features)
    fwiz_all = fwiz.lazy
    ### this saves the lazy transformer from featurewiz for next round ###

# Find the most common set of features (most stable) and use them to train a logistic regression model
common_features = list(set(selected_features[0]).intersection(*selected_features))
print('Common most stable features:', len(common_features), 'features are:\n', common_features)
#### Now transform your features to all-numeric using featurewiz' lazy transformer ###
X_train_selected_all = fwiz_all.transform(X_train)
X_test_selected_all = fwiz_all.transform(X_test)

# Evaluate the performance of the model on each round and compare it to the final accuracy with common features
accuracies = []
for i in range(num_rounds):
    model_round = LogisticRegression()
    model_round.fit(X_train_selected_all[selected_features[i]], y_train)
    accuracy_round = model_round.score(X_test_selected_all[selected_features[i]], y_test)
    accuracies.append(accuracy_round)
    
model_final = LogisticRegression()
model_final.fit(X_train_selected_all[common_features], y_train)
accuracy_final = model_final.score(X_test_selected_all[common_features], y_test)
print('Individual accuracy from',len(accuracies),'rounds is:',accuracies)
print('Average accuracy from 3 rounds = ', np.mean(accuracies), '\nvs. final accuracy with common features: ',accuracy_final)