import numpy as np
import pandas as pd
import random, re

class PreprocessingModule():
  def is_h1(self, column):
      pattern = re.compile("h1*")
      return bool(pattern.match(str(column)))

  def is_d1(self, column):
      pattern = re.compile("d1*")
      return bool(pattern.match(str(column)))

  def df_preprocessing(self, df, disable_h1=False, disable_d1=False, null_tolerance=100, onehot=False, 
    drop_columns=[], columns_to_keep=[]):
    if len(columns_to_keep) > 0:
        drop_columns = [column for column in df.columns if column not in columns_to_keep]
        df.drop(drop_columns, axis=1, inplace=True)
    else:
        # Dropping unused columns
        df.drop(drop_columns, axis = 1, inplace=True)

        # Dropping by null %, d1, h1
        for column in df.columns:
            total_nulls = df[column].isnull().sum()
            nulls_percentage = (total_nulls*100)/df.shape[0]
            if (disable_h1 and self.is_h1(column)) or (disable_d1 and self.is_d1(column)) or (null_tolerance <= nulls_percentage):
                df.drop(column, axis=1, inplace=True)

    # Filling by mean
    # columns= albumin_apache bilirubin_apache glucose_apache hematocrit_apache sodium_apache urineoutput_apache
    #          d1_diasbp_invasive_max d1_diasbp_invasive_min d1_diasbp_max d1_diasbp_min d1_diasbp_noninvasive_max d1_diasbp_noninvasive_min
    #          d1_mbp_invasive_max d1_mbp_invasive_min d1_mbp_noninvasive_max d1_mbp_noninvasive_min h1_mbp_invasive_max
    #          h1_mbp_invasive_min d1_albumin_max d1_albumin_min d1_bilirubin_max d1_bilirubin_min d1_calcium_max d1_calcium_min
    #          d1_glucose_max d1_glucose_min d1_hco3_max d1_hco3_min d1_hemaglobin_max d1_hemaglobin_min d1_hematocrit_max
    #          d1_hematocrit_min h1_albumin_max h1_albumin_min h1_bilirubin_max h1_bilirubin_min h1_calcium_max h1_calcium_min
    #          h1_glucose_max h1_glucose_min h1_hco3_max h1_hco3_min h1_hemaglobin_max h1_hemaglobin_min h1_hematocrit_max h1_hematocrit_min
    #          h1_sodium_max h1_sodium_min d1_arterial_pco2_max d1_arterial_pco2_min d1_arterial_ph_max d1_arterial_ph_min
    #          d1_arterial_po2_max d1_arterial_po2_min d1_pao2fio2ratio_max d1_pao2fio2ratio_min h1_arterial_pco2_max
    #          h1_arterial_pco2_min h1_arterial_ph_max h1_arterial_ph_min h1_arterial_po2_max h1_arterial_po2_min h1_pao2fio2ratio_max
    #          h1_pao2fio2ratio_min
    mean_columns = 'albumin_apache bilirubin_apache glucose_apache hematocrit_apache sodium_apache urineoutput_apache d1_diasbp_invasive_max d1_diasbp_invasive_min d1_diasbp_max d1_diasbp_min d1_diasbp_noninvasive_max d1_diasbp_noninvasive_min d1_mbp_invasive_max d1_mbp_invasive_min d1_mbp_noninvasive_max d1_mbp_noninvasive_min h1_mbp_invasive_max h1_mbp_invasive_min d1_albumin_max d1_albumin_min d1_bilirubin_max d1_bilirubin_min d1_calcium_max d1_calcium_min d1_glucose_max d1_glucose_min d1_hco3_max d1_hco3_min d1_hemaglobin_max d1_hemaglobin_min d1_hematocrit_max d1_hematocrit_min h1_albumin_max h1_albumin_min h1_bilirubin_max h1_bilirubin_min h1_calcium_max h1_calcium_min h1_glucose_max h1_glucose_min h1_hco3_max h1_hco3_min h1_hemaglobin_max h1_hemaglobin_min h1_hematocrit_max h1_hematocrit_min h1_sodium_max h1_sodium_min d1_arterial_pco2_max d1_arterial_pco2_min d1_arterial_ph_max d1_arterial_ph_min d1_arterial_po2_max d1_arterial_po2_min d1_pao2fio2ratio_max d1_pao2fio2ratio_min h1_arterial_pco2_max h1_arterial_pco2_min h1_arterial_ph_max h1_arterial_ph_min h1_arterial_po2_max h1_arterial_po2_min h1_pao2fio2ratio_max h1_pao2fio2ratio_min'.split(' ')
    for column in mean_columns:
      if column in df.columns:
        df[column].fillna((df[column].mean()), inplace=True)


    # Filling by mfv
    # columns: bun_apache creatinine_apache fio2_apache gcs_eyes_apache gcs_motor_apache gcs_unable_apache gcs_verbal_apache
    #          map_apache paco2_apache paco2_for_ph_apache paco2_for_ph_apache pao2_apache ph_apache temp_apache wbc_apache
    #          d1_heartrate_max d1_heartrate_min d1_heartrate_max d1_heartrate_min d1_resprate_max d1_resprate_min
    #          d1_spo2_max d1_spo2_min d1_sysbp_invasive_max d1_sysbp_invasive_min d1_sysbp_max d1_sysbp_min d1_sysbp_noninvasive_max
    #          d1_sysbp_noninvasive_min d1_temp_max d1_temp_min h1_diasbp_invasive_max h1_diasbp_invasive_min h1_diasbp_max
    #          h1_diasbp_min h1_diasbp_noninvasive_max h1_diasbp_noninvasive_min h1_heartrate_max h1_heartrate_min h1_mbp_max
    #          h1_mbp_min h1_mbp_noninvasive_max h1_mbp_noninvasive_min h1_resprate_max h1_resprate_min h1_spo2_max h1_spo2_min
    #          h1_sysbp_invasive_max h1_sysbp_invasive_min h1_sysbp_max h1_sysbp_min h1_sysbp_noninvasive_max h1_sysbp_noninvasive_min
    #          h1_temp_max h1_temp_min d1_bun_max d1_bun_min d1_creatinine_max d1_creatinine_min d1_inr_max d1_inr_min
    #          d1_lactate_max d1_lactate_min d1_platelets_max d1_platelets_min d1_potassium_max d1_potassium_min
    #          d1_sodium_max d1_sodium_min d1_wbc_max d1_wbc_min h1_bun_max h1_bun_min h1_creatinine_max h1_creatinine_min
    #          h1_inr_max h1_inr_min h1_lactate_max h1_lactate_min h1_platelets_max h1_platelets_min h1_potassium_max h1_potassium_min
    #          h1_wbc_max h1_wbc_min aids cirrhosis hepatic_failure immunosuppression leukemia lymphoma solid_tumor_with_metastasis
    mfv_columns = 'bun_apache creatinine_apache fio2_apache gcs_eyes_apache gcs_motor_apache gcs_unable_apache gcs_verbal_apache map_apache paco2_apache paco2_for_ph_apache paco2_for_ph_apache pao2_apache ph_apache heart_rate_apache temp_apache d1_mbp_max d1_mbp_min wbc_apache d1_heartrate_max d1_heartrate_min d1_heartrate_max d1_heartrate_min d1_resprate_max d1_resprate_min d1_spo2_max d1_spo2_min d1_sysbp_invasive_max d1_sysbp_invasive_min d1_sysbp_max d1_sysbp_min d1_sysbp_noninvasive_max d1_sysbp_noninvasive_min d1_temp_max d1_temp_min h1_diasbp_invasive_max h1_diasbp_invasive_min h1_diasbp_max h1_diasbp_min h1_diasbp_noninvasive_max h1_diasbp_noninvasive_min h1_heartrate_max h1_heartrate_min h1_mbp_max h1_mbp_min h1_mbp_noninvasive_max h1_mbp_noninvasive_min h1_resprate_max h1_resprate_min h1_spo2_max h1_spo2_min h1_sysbp_invasive_max h1_sysbp_invasive_min h1_sysbp_max h1_sysbp_min h1_sysbp_noninvasive_max h1_sysbp_noninvasive_min h1_temp_max h1_temp_min d1_bun_max d1_bun_min d1_creatinine_max d1_creatinine_min d1_inr_max d1_inr_min d1_lactate_max d1_lactate_min d1_platelets_max d1_platelets_min d1_potassium_max d1_potassium_min d1_sodium_max d1_sodium_min d1_wbc_max d1_wbc_min h1_bun_max h1_bun_min h1_creatinine_max h1_creatinine_min h1_inr_max h1_inr_min h1_lactate_max h1_lactate_min h1_platelets_max h1_platelets_min h1_potassium_max h1_potassium_min h1_wbc_max h1_wbc_min aids cirrhosis hepatic_failure immunosuppression leukemia lymphoma solid_tumor_with_metastasis'.split(' ')
    for column in mfv_columns:
      if column in df.columns:
        df[column].fillna((df[column].mode().iloc[0]), inplace=True)



    # Other fillings
    # columns: age bmi ethnicity gender apache_3j_diagnosis resprate_apache urineoutput_apache

    # Fill Null Age - proportion distribution?
    bins = pd.IntervalIndex.from_tuples([ (0, 20),(20, 40),(40, 60), (60,80), (80,100)])
    df['age_interval'] = pd.cut(df.age, bins)
    proportion = df['age_interval'].value_counts(normalize=True)
    df['age'] = df['age'].fillna(pd.Series(np.random.choice([15, 30, 50, 70, 90 ], p=proportion.values, size=len(df))))
    df = df.drop('age_interval', axis=1)

    # Fill bmi - mean by gender
    male_mean = df.groupby('gender')['bmi'].mean()['M']
    female_mean = df.groupby('gender')['bmi'].mean()['F']
    bmi_mean = df['bmi'].mean()
    conditions = [
      df['bmi'].notnull(),
      df['gender']=='M',
      df['gender']=='F',
    ]
    choices=[df['bmi'], male_mean, female_mean]
    df['bmi'] = df['bmi'].fillna(pd.Series(np.select(conditions, choices, default=bmi_mean)))
    male_mean = df.groupby('gender')['weight'].mean()['M']
    female_mean = df.groupby('gender')['weight'].mean()['F']
    conditions = [
      df['weight'].notnull(),
      df['gender']=='M',
      df['gender']=='F',
    ]
    choices=[df['weight'], male_mean, female_mean]
    weight_mean = df['bmi'].mean()
    df['weight'] = df['weight'].fillna(pd.Series(np.select(conditions, choices, default=weight_mean)))

    # Fill ethnicity - proportion distribution? + OneHot
    proportion = df['ethnicity'].value_counts(normalize=True)
    df['ethnicity'] = df['ethnicity'].fillna(pd.Series(np.random.choice(proportion.keys(), p=proportion.values, size=len(df))))

    # change OHE, change nulls for unknown

    # gender - fill by proportion
    df['gender'] = df['gender'].fillna(pd.Series(np.random.choice(['M', 'F'], p=[0.5, 0.5], size=len(df))))
    if (onehot):
        one_hot = pd.get_dummies(df['ethnicity'], drop_first=True)
        df = df.drop('ethnicity',axis = 1)
        df = df.join(one_hot)
        one_hot = pd.get_dummies(df['gender'], drop_first=True)
        df = df.drop('ethnicity',axis = 1)
        df = df.join(one_hot)
    else:
        df['ethnicity'] = df['ethnicity'].astype('category')
        df['gender'] = df['gender'].astype('category')

    # apache_3j_diagnosis - Normal Value in Range
    df['apache_3j_diagnosis'].fillna(random.randint(100, 200), inplace =True)

    # resprate_apache - Normal Value in Range
    df['resprate_apache'].fillna(random.randint(12, 16), inplace =True)

    # urineoutput_apache - Normal Value
    if 'urineoutput_apache' in df.columns:
      df['urineoutput_apache'].fillna(lambda x: random.randint(1200, 1800), inplace =True)
  
    # High correlations
    # bun_apache - Highly correlated (>0.9) removing d1, h1 ?
    # df = df.drop(['d1_bun_min', 'd1_bun_max', 'h1_bun_min', 'h1_bun_max'], axis=1)

    # creatinine_apache - Highly correlated (>0.9) removing d1, h1 ?
    # df = df.drop(['d1_creatinine_min', 'd1_creatinine_max', 'h1_creatinine_min', 'h1_creatinine_max'], axis=1)


    return df
