import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import hdbscan
import hdbscan.prediction as hdb_pred
from scipy import stats
import time
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def get_data(indir):
    train_df = pd.read_csv(os.path.abspath(indir) + os.path.sep + 'cc-transactions-project.csv')
    test_df = pd.read_csv(os.path.abspath(indir) + os.path.sep + 'cc-transactions-test-project.csv')
    return train_df, test_df


def format_data(df):
    # get local hour and time of day
    df['transaction_date_time'] = pd.to_datetime(df['transaction_date_time'])
    df['local_hour'] = df['transaction_date_time'].dt.hour
    df['time_of_day'] = pd.cut(df['local_hour'],
                               bins=[0, 5, 9, 17, 21, 24],
                               labels=['LateNight', 'Morning', 'WorkHours', 'Evening', 'Night'],
                               right=False)

    # get purchaser age at time of transaction
    df['transaction_date_time'] = pd.to_datetime(df['transaction_date_time'], utc=True)
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    current_year = pd.Timestamp('now').year
    df['age'] = current_year - df['dob'].dt.year

    # encode our categorical features
    le_job = LabelEncoder()
    le_category = LabelEncoder()
    le_gender = LabelEncoder()
    le_zip = LabelEncoder()
    df['job_encoded'] = le_job.fit_transform(df['job'].astype(str))
    df['category_encoded'] = le_category.fit_transform(df['category'].astype(str))
    df['gender_encoded'] = le_gender.fit_transform(df['gender'].astype(str))
    df['zip_encoded'] = le_zip.fit_transform(df['zip'].astype(str))

    # get time since last transaction per user
    df.sort_values(['cc_number', 'transaction_date_time'], inplace=True)
    df['time_diff'] = df.groupby('cc_number')['transaction_date_time'].diff().dt.total_seconds()
    df['time_diff'].fillna(df['time_diff'].median(), inplace=True)

    # log transform price
    df['log_amount'] = np.log1p(df['amount'])

    # features we will use for modeling
    features = ['log_amount',
                'age',
                'city_pop',
                'job_encoded',
                'category_encoded',
                'gender_encoded',
                'time_diff',
                'zip_encoded',
                'local_hour']
    X = df[features]
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled


def get_scores(df, X_scaled, state):
    # trained models
    models = {}

    # isolation forest
    print('running isolation forest model')
    start = time.time()
    iso = IsolationForest(contamination=0.05, random_state=state)
    iso.fit(X_scaled)
    df['iso_score'] = -iso.score_samples(X_scaled)
    models['iso'] = iso
    print(f'\ttraining time: {time.time() - start:.2f} seconds')

    # svm
    print('running one-class svm model')
    start = time.time()
    ocsvm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
    df['ocsvm_score'] = -ocsvm.fit(X_scaled).decision_function(X_scaled)
    models['svm'] = ocsvm
    print(f'\ttraining time: {time.time() - start:.2f} seconds')

    # local outlier
    print('running local outlier factor model')
    start = time.time()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1)
    lof.fit(X_scaled)
    df['lof_score'] = -lof.decision_function(X_scaled)
    models['lof'] = lof
    print(f'\ttraining time: {time.time() - start:.2f} seconds')

    # hdbscan inlier / outlier
    # we can train and use for predict
    print('running hdbscan model')
    start = time.time()
    hdb = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
    hdb.fit(X_scaled)
    df['hdbscan_score'] = (hdb.labels_ == -1).astype(int)
    # unused
    # df['hdbscan_label'] = hdb.labels_
    models['hdbscan'] = hdb
    print(f'\ttraining time: {time.time() - start:.2f} seconds')

    # zscore on log amount 
    print('getting zscores')
    start = time.time()
    df['z_score'] = np.abs(stats.zscore(df['log_amount']))
    print(f'\ttraining time: {time.time() - start:.2f} seconds')
    
    return models


def get_preds(models, test_df, X_scaled_test):
    print('generating predictions on test data')

    # isof predictions
    if 'iso' in models:
        start = time.time()
        print('running isof on test data')
        test_df['iso_score'] = -models['iso'].score_samples(X_scaled_test)
        print(f'\ttime: {time.time() - start:.2f} seconds')

    # svm predictions
    if 'svm' in models:
        print('running svm on test data')
        test_df['ocsvm_score'] = -models['svm'].decision_function(X_scaled_test)
        print(f'\ttime: {time.time() - start:.2f} seconds')

    # lof predictions
    if 'lof' in models:
        start = time.time()
        print('running lof on test data')
        test_df['lof_score'] = -models['lof'].decision_function(X_scaled_test)
        print(f'\ttime: {time.time() - start:.2f} seconds')

    # hdbscan predictions
    if 'hdbscan' in models:
        start = time.time()
        print('running hdbscan on test data')
        hdb_labels, hdb_probs = hdb_pred.approximate_predict(models['hdbscan'],
                                                             X_scaled_test)
        test_df['hdbscan_outlier'] = (hdb_labels == -1).astype(int)
        test_df['hdbscan_score'] = 1.0 - hdb_probs
        print(f'\ttime: {time.time() - start:.2f} seconds')

    # zscore on log_amount
    start = time.time()
    print('running zscore on test data')
    test_df['z_score'] = np.abs(stats.zscore(test_df['log_amount']))
    print(f'\ttime: {time.time() - start:.2f} seconds')

    # normalize our ensemble scores
    norm_ensemble(test_df)

    return test_df


def age_fraud_score(age_series):
    # from 78 to 100 = 23 years
    # avg lifetime is ~77 
    max_extra = 23
    scores = pd.Series(0.0, index=age_series.index)

    # underage is fraud
    scores[age_series < 18] = 1.0

    # scale score when over life expectancy
    over_mask = age_series > 77
    scores[over_mask] = np.minimum((age_series[over_mask] - 77) / max_extra, 1.0)
    return scores


def norm_ensemble(df):
    # get age values first
    df['age_score'] = age_fraud_score(df['age'])

    # normalize and ensemble all of our scores
    scores = ['iso_score',
              'ocsvm_score',
              'lof_score',
              'z_score',
              'hdbscan_score',
              'age_score']

    # normalize the scores
    for score in scores:
        df[score] = (df[score] - df[score].min()) / (df[score].max() - df[score].min())

    # get the mean ensemble score per entry
    df['ensemble_score'] = df[scores].mean(axis=1)
    return


def main(indir, outdir):
    ## model training / testing values
    # quantile threshold
    quant = 0.99
    # sate for random sampling
    state = 4444
    # 30% of training data to use for modeling
    sample_frac = 0.1

    # get our datasets
    train, test = get_data(indir)
    
    # format the data and generate new features
    ftrain = format_data(train)
    ftest = format_data(test)

    # get our random samples of the training data
    rng = np.random.RandomState(state)
    sampled_indices = rng.choice(ftrain.shape[0],
                                 size=int(sample_frac * ftrain.shape[0]),
                                 replace=False)

    # train our models and get predictions    
    ftrain_sampled = ftrain[sampled_indices]
    train_sampled = train.iloc[sampled_indices]
    print(f'training on: {train_sampled.shape[0]} samples')

    # get scores on the training data
    models = get_scores(train_sampled, ftrain_sampled, state)

    # normalize and ensemble our model predictions
    norm_ensemble(train_sampled)

    # apply threshold for identifying fraudlient activity
    # on the training data
    threshold = train_sampled['ensemble_score'].quantile(quant)
    train_sampled['is_fraud'] = (train_sampled['ensemble_score'] >= threshold).astype(int)

    # run models on test data
    print(f'\ntesting on: {test.shape[0]} samples')
    test_results = get_preds(models, test, ftest)

    # apply threshold to test data
    threshold = test_results['ensemble_score'].quantile(quant)
    test_results['is_fraud'] = (test_results['ensemble_score'] > threshold).astype(int)

    # save out our results
    results = test_results[['record_number', 'is_fraud']]
    results.to_csv(os.path.abspath(outdir) + os.path.sep + 'output.csv', index=False)
    
    # show distribution of ensemble score
    plt.hist(test_results['ensemble_score'], bins=50)
    plt.title('Distribution of Ensemble Scores')
    plt.savefig(os.path.abspath(outdir) + os.path.sep + 'output.png')

    return


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--indir', required=True, help='Input directory that contains .csv files')
    ap.add_argument('-o', '--outdir', required=True, help='Output directory to save results files')
    args = ap.parse_args()
    main(args.indir, args.outdir)