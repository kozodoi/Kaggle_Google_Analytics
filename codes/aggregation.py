import pandas as pd
import numpy as np
import scipy.stats


### DATA AGGREGATION

def aggregate_data(df, 
                   group_var, 
                   num_stats = ['mean', 'sum'], 
                   label     = None, 
                   sd_zeros  = False):
    
    '''
    Aggregate data
    '''
    
    ### SEPARATE FEATURES
    
    # display info
    print("- Preparing the dataset...")

    # find factors
    df_factors = [f for f in df.columns if df[f].dtype == "object"]
    df_factors = ['fullVisitorId', 'device_operatingSystem', 'geoNetwork_country', 'channelGrouping']
        
    # partition subsets
    if type(group_var) == str:
        num_df = df[[group_var] + list(set(df.columns) - set(df_factors))]
        fac_df = df[df_factors]
    else:
        num_df = df[group_var + list(set(df.columns) - set(df_factors))]
        fac_df = df[df_factors]      
    
    # display info
    num_facs = fac_df.shape[1] - 1
    num_nums = num_df.shape[1] - 1
    print("- Extracted %.0f factors and %.0f numerics..." % (num_facs, num_nums))


    ##### AGGREGATION

    # aggregate numerics
    if (num_nums > 0):
        print("- Aggregating numeric features...")
        if type(group_var) == str:
            num_df = num_df.groupby([group_var]).agg(num_stats)
            num_df.columns = ["_".join(col).strip() for col in num_df.columns.values]
            num_df = num_df.sort_index()
        else:
            num_df = num_df.groupby(group_var).agg(num_stats)
            num_df.columns = ["_".join(col).strip() for col in num_df.columns.values]
            num_df = num_df.sort_index()

    # aggregate factors
    if (num_facs > 0):
        print("- Aggregating factor features...")
        if type(group_var) == str:
            fac_df = fac_df.groupby([group_var]).agg([("mode", lambda x: scipy.stats.mode(x)[0][0])])
            fac_df.columns = ["_".join(col).strip() for col in fac_df.columns.values]
            fac_df = fac_df.sort_index()
        else:
            fac_df = fac_df.groupby(group_var).agg([("mode", lambda x: scipy.stats.mode(x)[0][0])])
            fac_df.columns = ["_".join(col).strip() for col in fac_df.columns.values]
            fac_df = fac_df.sort_index()


    ##### MERGER

    # merge numerics and factors
    if ((num_facs > 0) & (num_nums > 0)):
        agg_df = pd.concat([num_df, fac_df], axis = 1)
    
    # use factors only
    if ((num_facs > 0) & (num_nums == 0)):
        agg_df = fac_df
        
    # use numerics only
    if ((num_facs == 0) & (num_nums > 0)):
        agg_df = num_df
        

    ##### LAST STEPS

    # update labels
    if (label != None):
        agg_df.columns = [label + "_" + str(col) for col in agg_df.columns]
    
    # impute zeros for SD
    if (sd_zeros == True):
        stdevs = agg_df.filter(like = "_std").columns
        for var in stdevs:
            agg_df[var].fillna(0, inplace = True)
            
    # dataset
    agg_df = agg_df.reset_index()
    print("- Final dimensions:", agg_df.shape)
    return agg_df