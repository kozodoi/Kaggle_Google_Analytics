import pandas as pd
import numpy as np

from pandas.io.json import json_normalize
import json
from ast import literal_eval



### IMPORT CSV WITH JSON

def read_csv_with_json(path, 
                       json_cols, 
                       nrows = None):
    
    '''
    Import CSV with JSON columns
    '''
        
    # import data frame
    df = pd.read_csv(path, 
                     converters = {column: json.loads for column in json_cols}, 
                     dtype = {'fullVisitorId': 'str'},
                     nrows = nrows)
    
    # extract values
    for column in json_cols:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis = 1).merge(column_as_df, right_index = True, left_index = True)

    # return data
    print(f"Loaded {os.path.basename(path)}: {df.shape}")
    return df



### ADD CUSTOM DIMENSIONS

def add_custom_dim(df):
    
    '''
    Unfold custom dimensions
    '''

    # extract custom dimensions
    df['customDimensions'] = df['customDimensions'].apply(literal_eval)
    df['customDimensions'] = df['customDimensions'].str[0]
    df['customDimensions'] = df['customDimensions'].apply(lambda x: {'index':np.NaN,'value':np.NaN} if pd.isnull(x) else x)

    column_as_df = json_normalize(df['customDimensions'])
    column_as_df.columns = [f"customDimensions_{subcolumn}" for subcolumn in column_as_df.columns]
    df = df.drop('customDimensions', axis=1).merge(column_as_df, right_index = True, left_index = True)
    
    return df



### FILL MISSINGS

def fill_na(df):
    
    '''
    Fill missing values
    '''
    
    ##### IMPUTE NA DIFFERENTLY
    
    # NA = unknown
    to_NA_cols = ['trafficSource_adContent',
                  'trafficSource_adwordsClickInfo.adNetworkType',
                  'trafficSource_adwordsClickInfo.slot',
                  'trafficSource_adwordsClickInfo.gclId',
                  'trafficSource_keyword',
                  'trafficSource_referralPath',
                  'customDimensions_value']

    # NA = zero
    to_0_cols = ['totals_transactionRevenue',
                 'trafficSource_adwordsClickInfo.page',
                 'totals_sessionQualityDim','totals_bounces',
                 'totals_timeOnSite',
                 'totals_newVisits',
                 'totals_pageviews',
                 'customDimensions_index',
                 'totals_transactions',
                 'totals_totalTransactionRevenue']

    # NA = TRUE / FALSE
    to_true_cols  = ['trafficSource_adwordsClickInfo.isVideoAd']
    to_false_cols = ['trafficSource_isTrueDirect']
    
    # impute missings
    df[to_NA_cols]    = df[to_NA_cols].fillna('NA')
    df[to_0_cols]     = df[to_0_cols].fillna(0)
    df[to_true_cols]  = df[to_true_cols].fillna(True)
    df[to_false_cols] = df[to_false_cols].fillna(False)
    
    
    
    ##### REPLACE SOME LEVELS WITH NA
    
    # not available, not provided, etc.
    cols_to_replace = {
        'socialEngagementType' : 'Not Socially Engaged',
        'device_browserSize' : 'not available in demo dataset', 
        'device_flashVersion' : 'not available in demo dataset', 
        'device_browserVersion' : 'not available in demo dataset', 
        'device_language' : 'not available in demo dataset',
        'device_mobileDeviceBranding' : 'not available in demo dataset',
        'device_mobileDeviceInfo' : 'not available in demo dataset',
        'device_mobileDeviceMarketingName' : 'not available in demo dataset',
        'device_mobileDeviceModel' : 'not available in demo dataset',
        'device_mobileInputSelector' : 'not available in demo dataset',
        'device_operatingSystemVersion' : 'not available in demo dataset',
        'device_screenColors' : 'not available in demo dataset',
        'device_screenResolution' : 'not available in demo dataset',
        'geoNetwork_city' : 'not available in demo dataset',
        'geoNetwork_cityId' : 'not available in demo dataset',
        'geoNetwork_latitude' : 'not available in demo dataset',
        'geoNetwork_longitude' : 'not available in demo dataset',
        'geoNetwork_metro' : ['not available in demo dataset', '(not set)'], 
        'geoNetwork_networkDomain' : ['unknown.unknown', '(not set)'], 
        'geoNetwork_networkLocation' : 'not available in demo dataset',
        'geoNetwork_region' : 'not available in demo dataset',
        'trafficSource_adwordsClickInfo.criteriaParameters' : 'not available in demo dataset',
        'trafficSource_campaign' : '(not set)', 
        'trafficSource_keyword' : ['(not provided)', '(not set)'], 
        'networkDomain': '(not set)', 
        'city': '(not set)'
    }
    df = df.replace(cols_to_replace,'NA')
    
    return df



### ENCODE FACTORS

def encode_factors(df, 
                   method = "label"):
    
    '''
    Encode factor variables
    '''
    
    # label encoding
    if method == "label":
        factors = [f for f in df.columns if df[f].dtype == "object"]
        for var in factors:
            df[var], _ = pd.factorize(df[var])
        
    # dummy encoding
    if method == "dummy":
        df = pd.get_dummies(df, drop_first = True)
    
    # dataset
    return df