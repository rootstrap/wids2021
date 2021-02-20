from sklearn.model_selection import train_test_split
import pandas as pd 

class PredictingModule():  
  def split_data(self, df, test_size=0.2, val_size=0.25):
    X = df.drop(['diabetes_mellitus','encounter_id'], axis=1)
    y = pd.to_numeric(df['diabetes_mellitus'].values)
    X = pd.DataFrame(X, columns=df.drop(['diabetes_mellitus','encounter_id'], axis=1).columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=123)
    if (val_size!=0):
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=123)
    else:
        X_val = []
        y_val = []

    return X_train, X_val, X_test, y_train, y_val, y_test

  def predict_probabilities(self, model, X_train, X_val, y_train, y_val, X_test, y_test):
    y_train_pred = model.predict_proba(X_train)
    y_pred = model.predict_proba(X_test)
    y_val_pred = model.predict_proba(X_val)

    df_train_pred = pd.DataFrame()
    df_train_pred['probability'] = y_train_pred[:,1]
    df_train_pred['predicted_value'] = df_train_pred['probability'].apply(lambda x: 1 if x>0.5 else 0)
    df_train_pred['real_value'] = y_train

    df_val_pred = pd.DataFrame()
    df_val_pred['probability'] = y_val_pred[:,1]
    df_val_pred['predicted_value'] = df_val_pred['probability'].apply(lambda x: 1 if x>0.5 else 0)
    df_val_pred['real_value'] = y_val

    df_test_pred = pd.DataFrame()
    df_test_pred['probability'] = y_pred[:,1]
    df_test_pred['predicted_value'] = df_test_pred['probability'].apply(lambda x: 1 if x>0.5 else 0)
    df_test_pred['real_value'] = y_test

    return df_train_pred, df_val_pred, df_test_pred
