# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re


class PreprocessingModule2():

    def get_min_max(self, df):
        cols = ''
        for c in df.columns:
            if bool(re.compile('.*_min').match(str(c))) or bool(re.compile('.*_max').match(str(c))):
                cols = cols + ' ' + c
        return cols

    def is_h1(self, column):
        pattern = re.compile('h1*')
        return bool(pattern.match(str(column)))

    def is_d1(self, column):
        pattern = re.compile('d1*')
        return bool(pattern.match(str(column)))

    def transform_one_hot(self, df, cols):
        cat_cols = cols[:]
        for c in cat_cols:
            is_binary = False
            if len(df[c].unique()) == 2:
                unique_vars = df[c].unique() == [0, 1]
                is_binary = unique_vars[0] and unique_vars[1]
            if not is_binary:
                one_hot = pd.get_dummies(df[c], prefix=c, drop_first=True)
                df.drop(c, axis=1, inplace=True)
                df = pd.concat([df, one_hot], axis=1)
                cols.remove(c)
                cols.extend(one_hot.columns)
        return df, cols

    def transform_cols(self, df, categorical_cols):
        for c in df.columns:
            if c in categorical_cols:
                df[c] = df[c].astype('category')
            else:
                df[c] = pd.to_numeric(df[c])
        return df

    def transform_age(self, df):
        df['abmi'] = df['age'] / df['bmi']
        df['abmi'].fillna(0, inplace=True)

        df['agi'] = df['weight'] / df['age']
        df['agi'].replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        df['age_range'] = df['age'].apply(lambda x: (24 if x >= 85 else (16 if x < 85 and x >= 75 else (
            13 if x < 75 and x >= 65 else (11 if x < 64 and x >= 60 else (5 if x < 60 and x >= 45 else 0))))))

        return df

    def mean_by_gender(self, df, cols):
        for c in cols:
            male_mean = df.groupby('gender')[c].mean()['M']
            female_mean = df.groupby('gender')[c].mean()['F']
            conditions = [df[c].notnull(), df['gender'] == 'M',
                          df['gender'] == 'F']
            choices = [df[c], male_mean, female_mean]
            height_mean = df[c].mean()
            df[c] = df[c].fillna(pd.Series(np.select(conditions,
                                                     choices, default=height_mean)))
        return df

    def fill_mean(self, df, cols):
        for column in cols.split(' '):
            if column in df.columns:
                df[column].fillna(df[column].mean(), inplace=True)

        return df

    def comorbidity(self, df):
        df['comorbidity_score'] = df['cirrhosis'] * 4 \
                                  + df['immunosuppression'] * 10 + df['leukemia'] * 10 \
                                  + df['solid_tumor_with_metastasis'] * 11 + df['lymphoma'
                                  ] * 13 + df['hepatic_failure'] * 16 + df['aids'] \
                                  * 23

        df.drop([
            'cirrhosis',
            'immunosuppression',
            'leukemia',
            'solid_tumor_with_metastasis',
            'lymphoma',
            'hepatic_failure',
            'aids',
        ], axis=1, inplace=True)
        return df

    def gcs(self, df):
        df['gcs'] = df['gcs_eyes_apache'] + df['gcs_motor_apache'] \
                    + df['gcs_verbal_apache'] + df['gcs_unable_apache']
        df.drop(['gcs_eyes_apache', 'gcs_motor_apache',
                 'gcs_verbal_apache', 'gcs_unable_apache'], axis=1, inplace=True)
        df['gcs'].fillna(0, inplace=True)
        return df

    def mbp(self, df):
        df['mbp'] = ((df['d1_mbp_invasive_max'] == df['d1_mbp_max'
        ]) & (df['d1_mbp_noninvasive_max']
              == df['d1_mbp_invasive_max'])
                     | (df['d1_mbp_invasive_min'] == df['d1_mbp_min'
                ]) & (df['d1_mbp_noninvasive_min']
                      == df['d1_mbp_invasive_min'])
                     | (df['h1_mbp_invasive_max'] == df['h1_mbp_max'
                ]) & (df['h1_mbp_noninvasive_max']
                      == df['h1_mbp_invasive_max'])
                     | (df['h1_mbp_invasive_min'] == df['h1_mbp_min'
                ]) & (df['h1_mbp_noninvasive_min']
                      == df['h1_mbp_invasive_min'])).astype(np.int8)
        return df

    def sysbp(self, df):
        df['sysbp'] = ((df['d1_sysbp_invasive_max']
                        == df['d1_sysbp_max'])
                       & (df['d1_sysbp_noninvasive_max']
                          == df['d1_sysbp_invasive_max'])
                       | (df['d1_sysbp_invasive_min']
                          == df['d1_sysbp_min'])
                       & (df['d1_sysbp_noninvasive_min']
                          == df['d1_sysbp_invasive_min'])
                       | (df['h1_sysbp_invasive_max']
                          == df['h1_sysbp_max'])
                       & (df['h1_sysbp_noninvasive_max']
                          == df['h1_sysbp_invasive_max'])
                       | (df['h1_sysbp_invasive_min']
                          == df['h1_sysbp_min'])
                       & (df['h1_sysbp_noninvasive_min']
                          == df['h1_sysbp_invasive_min'
                          ])).astype(np.int8)
        return df

    def create_invnoninv_diffs(self, df):
        df['d1_mbp_invnoninv_max_diff'] = df['d1_mbp_invasive_max'] \
                                          - df['d1_mbp_noninvasive_max']
        df['h1_mbp_invnoninv_max_diff'] = df['h1_mbp_invasive_max'] \
                                          - df['h1_mbp_noninvasive_max']
        df['d1_mbp_invnoninv_min_diff'] = df['d1_mbp_invasive_min'] \
                                          - df['d1_mbp_noninvasive_min']
        df['h1_mbp_invnoninv_min_diff'] = df['h1_mbp_invasive_min'] \
                                          - df['h1_mbp_noninvasive_min']
        df['d1_diasbp_invnoninv_max_diff'] = df['d1_diasbp_invasive_max'] \
                                             - df['d1_diasbp_noninvasive_max']
        df['h1_diasbp_invnoninv_max_diff'] = df['h1_diasbp_invasive_max'] \
                                             - df['h1_diasbp_noninvasive_max']
        df['d1_diasbp_invnoninv_min_diff'] = df['d1_diasbp_invasive_min'] \
                                             - df['d1_diasbp_noninvasive_min']
        df['h1_diasbp_invnoninv_min_diff'] = df['h1_diasbp_invasive_min'] \
                                             - df['h1_diasbp_noninvasive_min']
        df['d1_sysbp_invnoninv_max_diff'] = df['d1_sysbp_invasive_max'] \
                                            - df['d1_sysbp_noninvasive_max']
        df['h1_sysbp_invnoninv_max_diff'] = df['h1_sysbp_invasive_max'] \
                                            - df['h1_sysbp_noninvasive_max']
        df['d1_sysbp_invnoninv_min_diff'] = df['d1_sysbp_invasive_min'] \
                                            - df['d1_sysbp_noninvasive_min']
        df['h1_sysbp_invnoninv_min_diff'] = df['h1_sysbp_invasive_min'] \
                                            - df['h1_sysbp_noninvasive_min']

        return df

    def remove_uniqueness(self, df):
        for c in df.columns:
            if len(df[c].unique()) == 1:
                df.drop(c, axis=1, inplace=True)
        return df

    def check_nulls(self, df, drop_cols, null_tolerance, disable_h1, disable_d1):
        for column in df.columns:
            total_nulls = df[column].isnull().sum()
            nulls_percentage = (total_nulls * 100) / df.shape[0]
            if (disable_h1 and self.is_h1(column)) or (disable_d1 and self.is_d1(column)) or (
                    null_tolerance <= nulls_percentage):
                drop_cols.append(column)

        return drop_cols

    def drop_df_columns(self, df, drop_cols, columns_to_keep):
        drop_cols = [column for column in drop_cols if column in df.columns]
        if len(columns_to_keep) > 0:
            drop_cols = [column for column in drop_cols if column not in columns_to_keep]

        df.drop(drop_cols, axis=1, inplace=True)
        return df

    def preprocessing(self, df, disable_h1=False, disable_d1=False, null_tolerance=100, onehot=False,
                      drop_columns=[], columns_to_keep=[]):
        categorical_cols = []

        drop_columns = self.check_nulls(df, drop_columns, null_tolerance, disable_h1, disable_d1)
        df = self.remove_uniqueness(df)

        df = self.transform_age(df)
        drop_columns.append('age')

        categorical_cols.append('elective_surgery')

        df['ethnicity'].fillna('Unkown', inplace=True)
        categorical_cols.append('ethnicity')

        df = self.mean_by_gender(df, ['height', 'weight', 'bmi'])
        df['gender'] = df['gender'].fillna(pd.Series(np.random.choice(['M', 'F'], p=[0.5, 0.5], size=len(df))))
        df['gender'] = df['gender'].apply(lambda x: 0 if x == 'M' else 1)

        df['apache_3j_diagnosis'].fillna(-100, inplace=True)
        df['apache_3j_diagnosis'] = round(df['apache_3j_diagnosis'])
        categorical_cols.append('apache_3j_diagnosis')

        df['apache_post_operative'].fillna(0, inplace=True)
        categorical_cols.append('apache_post_operative')

        df['apache_2_diagnosis'] = df['apache_2_diagnosis'].fillna(
            pd.Series(np.random.choice([110, 210, 300], p=[0.55, 0.05, 0.4], size=len(df))))
        df['apache_2_diagnosis'] = round(df['apache_2_diagnosis'])
        categorical_cols.append('apache_2_diagnosis')

        df['arf_apache'].fillna(0, inplace=True)
        categorical_cols.append('arf_apache')

        mean_columns = 'albumin_apache bilirubin_apache bun_apache creatinine_apache fio2_apache glucose_apache heart_rate_apache hematocrit_apache intubated_apache map_apache paco2_apache paco2_for_ph_apache pao2_apache ph_apache resprate_apache sodium_apache temp_apache urineoutput_apache ventilated_apache wbc_apache h1_glucose_min'

        mean_columns = mean_columns + ' ' + self.get_min_max(df)
        df = self.fill_mean(df, mean_columns)

        categorical_cols.append('ventilated_apache')

        df = self.comorbidity(df)
        df = self.gcs(df)
        df = self.mbp(df)
        df = self.sysbp(df)
        df = self.create_invnoninv_diffs(df)
        df = self.drop_df_columns(df, drop_columns, columns_to_keep)

        df = self.transform_cols(df, categorical_cols)
        if onehot:
            df, categorical_cols = self.transform_one_hot(df, categorical_cols)

        return df, categorical_cols


