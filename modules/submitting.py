import pandas as pd

class SubmittingModule():
  def generate_submit_result(self, test_data, model, file_name):
    y_pred_submission = model.predict_proba(test_data.drop('encounter_id', axis=1))
    df_final = pd.DataFrame(columns=['encounter_id', 'diabetes_mellitus'])
    df_final['encounter_id'] = test_data['encounter_id'].astype('int32')
    df_final['diabetes_mellitus'] = y_pred_submission[:,1]
    print(df_final.dtypes)
    df_final.to_csv(file_name, index=False)
