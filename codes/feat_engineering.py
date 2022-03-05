import pandas as pd
import numpy as np
import scipy.stats
from aggregation import aggregate_data


### CREATE MONTH FEATURE

def add_months_to_end(df, 
                      months = 12):
    
    '''
    Add month indicator
    '''

    df['months_to_end'] = np.nan

    for t in range(months):
        t_min = df.date.max() - pd.DateOffset(months = (t + 1))
        t_max = df.date.max() - pd.DateOffset(months = t)
        df.loc[(df.date > t_min) & (df.date <= t_max), 'months_to_end'] = t + 1
        
    return df



### CREATE DATE FEATURES

def encode_date(df):
    
    '''
    Add date features
    '''
    
    attrs = [
             #'Year', 'Month', 'Week', 'Day', 
             'Dayofweek', 
             #'Dayofyear',
             #'Is_month_end', 'Is_month_start', 
             #'Is_quarter_end', 'Is_quarter_start', 
             #'Is_year_end', 'Is_year_start'
            ]
        
    for attr in attrs:
        df['date_' + attr] = getattr(df['date'].dt, attr.lower())
            
    return df



### WRAPPER FEATURE ENGINEERING FUNCTION

def create_data(df, 
                x_idx,
                y_idx, 
                old_months = 7, 
                agg_months = 1):
    
    '''
    Create features
    '''

    ###### DATA PARTITIONING

    # extract X 
    x = df.iloc[x_idx]
    ids = x.fullVisitorId.unique()
    
    # append previous months to train
    old_x = x.date.min() - pd.DateOffset(months = old_months)
    old_x = df[(df.date >= old_x) & (df.date < x.date.min()) & (df.fullVisitorId.isin(ids))]
    x = pd.concat([x, old_x], axis = 0)
    
    # extract Y
    if len(y_idx) > 0:
        
        y = df.iloc[y_idx][['fullVisitorId', 'totals_transactionRevenue']]
        y['target'] = np.log1p(y.groupby('fullVisitorId').totals_transactionRevenue.transform('sum'))
        y = y[['fullVisitorId', 'target']]
        y.drop_duplicates(inplace = True)
        y = y.loc[y.fullVisitorId.isin(ids)]

        
    
    ###### FEATURE ENGINEERING
    
    ### AGGREGATIONS

    # aggregations (total)
    grp_x = aggregate_data(x, group_var = 'fullVisitorId')
    
    '''
    # aggregations (monthly)
    x = add_months_to_end(x, months = agg_months)
    for t in range(agg_months):
        tmp_x = x.loc[x.months_to_end <= (t + 1)]
        tmp_grp_x = aggregate_data(tmp_x, group_var = 'fullVisitorId', label = 'm' + str(t+1))
        grp_x     =  grp_x.merge(tmp_grp_x, how = 'left', on = 'fullVisitorId')      
    '''

    
    ### OTHER VARIABLES
    
    # number of visits
    drops = list(grp_x.filter(like = 'visitNumber').columns)
    for var in drops:
        del grp_x[var]
    x['num_visits'] = x.groupby('fullVisitorId').visitNumber.transform('max')
    tmp = x[['fullVisitorId', 'num_visits']].drop_duplicates()
    grp_x = grp_x.merge(tmp, how = 'left', on = 'fullVisitorId')
    
    # number of paying visits
    #x_tmp = x[x.totals_transactionRevenue > 0]
    #x_tmp['num_paying_visits'] = x_tmp.groupby('fullVisitorId').visitId.transform('count')
    #x_tmp = x_tmp[['fullVisitorId', 'num_paying_visits']].drop_duplicates()
    #grp_x = grp_x.merge(x_tmp, how = 'left', on = 'fullVisitorId')
    
    # add recency
    x['recency'] = x.groupby('fullVisitorId').date.transform('max')
    x['recency'] = ((x.date.max() - x['recency']) / np.timedelta64(1, 'D')).astype(int)
    tmp = x[['fullVisitorId', 'recency']].drop_duplicates()
    grp_x = grp_x.merge(tmp, how = 'left', on = 'fullVisitorId')

    # add frequency
    x['frequency'] = x.groupby('fullVisitorId').date.transform('count')
    tmp = x[['fullVisitorId', 'frequency']].drop_duplicates()
    grp_x = grp_x.merge(tmp, how = 'left', on = 'fullVisitorId')
    grp_x['frequency'].fillna(0, inplace = True)
    
    # day of the week
    #tmp = encode_date(x)
    #tmp = tmp[['fullVisitorId', 'date_Dayofweek']]
    #tmp = tmp.groupby('fullVisitorId').agg([("mode", lambda x: scipy.stats.mode(x)[0][0])])
    #tmp.columns = ["_".join(col).strip() for col in tmp.columns.values]
    #tmp = tmp.sort_index()
    #grp_x = grp_x.merge(tmp, how = 'left', on = 'fullVisitorId')
    
    
    ###### ALIGNMENT
    
    if len(y_idx) > 0:
    
        # merge zeros for new ids
        new_ids = list(set(grp_x.fullVisitorId) - set(y.fullVisitorId))
        new_ids = grp_x[grp_x.fullVisitorId.isin(new_ids)][['fullVisitorId']]
        y = y.merge(new_ids, how = 'outer', on = 'fullVisitorId')
        y.fillna(0, inplace = True)

        # align X and Y
        y = y.sort_values('fullVisitorId')['target'].reset_index(drop = True)
        x = grp_x.sort_values('fullVisitorId').reset_index(drop = True)
        return x, y
    
    else:
        
        x = grp_x.sort_values('fullVisitorId').reset_index(drop = True)
        return x