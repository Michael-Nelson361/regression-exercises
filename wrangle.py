
# import basic libraries
import pandas as pd
import numpy as np
import os
import env

# import some more specialized libraries
import tqdm
from scipy import stats
from sklearn.model_selection import train_test_split
        

def df_info(df,include=False,samples=1):
    """
    Function takes a dataframe and returns potentially relevant information about it (including a sample)

    include=bool, default to False. To add the results from a describe method, pass True to the argument.
    samples=int, default to 1. Shows 1 sample by default, but can be modified to include more samples if desired.
    """
    
    # create the df_inf dataframe
    df_inf = pd.DataFrame(index=df.columns,
            data = {
                'nunique':df.nunique()
                ,'dtypes':df.dtypes
                ,'isnull':df.isnull().sum()
            })
    
    # append samples based on input
    if samples >= 1:
        df_inf = df_inf.merge(df.sample(samples).iloc[0:samples].T,how='left',left_index=True,right_index=True)
    
    # append describe results if option selected
    if include == True:
        return df_inf.merge(df.describe(include='all').T,how='left',left_index=True,right_index=True)
    elif include == False:
        return df_inf
    else:
        print('Value passed to "include" argument is invalid.')
        

def print_libs():
    """
    Function that prints all libraries used up to present. Takes no arguments and returns none.
    """
    libraries = [
        'import itertools -> iterations',
        'from tqdm import tqdm -> progress bars on for loops'
        'import pandas as pd -> large scale database work',
        'import numpy as np -> advanced numerical work',
        'import matplotlib.pyplot as plt -> plotting work',
        'import seaborn as sns -> advanced and intuitive plotting',
        'from scipy import stats -> statistical work',
        'from pydataset import data -> list of datasets',
        'import os -> operating system work',
        'import warnings -> getting rid of pesky warnings',
        'from sklearn import metrics -> model metrics',
        'from sklearn.impute import SimpleImputer -> dynamic value filling',
        'from sklearn.model_selection import train_test_split -> splitting datasets',
        'from sklearn.tree import DecisionTreeClassifier, plot_tree -> DT modeling',
        'from sklearn.neighbors import KNeighborsClassifier -> KNN modeling',
        'from sklearn.ensemble import RandomForestClassifier -> RF modeling',
        'from sklearn.linear_model import LogisticRegression -> LR modeling'
    ]
    
    for library in libraries:
        print(library)
        

# Generic function to check if a file exists
def check_file_exists(filename,query,url):
    """
    Function takes a filename, query, and url and checks if the file exists. It will load the dataset requested from either SQL or from the local file.
    """
    if os.path.exists(filename):
        print('Reading from file...')
        df = pd.read_csv(filename,index_col=0)
    else:
        print('Reading from database...')
        df = pd.read_sql(query,url)
        
        df.to_csv(filename)
    
    return df
        

def drop_extras(df,target,degree=6):
    """
    Function to drop extra columns that may have a smaller impact on the model. Requires dataframe be cleaned first.
    
    Takes a DataFrame, and returns a DataFrame.
    
    Degree indicates the index of object columns to begin selecting for drop off.
        Hint: smaller value means drop more columns, larger value means drop fewer columns!
    """
    from scipy import stats
    
    corr_dict = {}
    obj_cols = []
    alpha = 0.05
    
    # grab object columns from dataframe
    for col in df.columns:
        if df[col].dtype == 'O':
            # print(f'{col}: object')
            obj_cols.append(col)
    
    # get p-values of columns
    for col in obj_cols:
        observed = pd.crosstab(df[target],df[col])
        # print(observed)

        t,p,dof,expected = stats.chi2_contingency(observed)

        # if p < alpha:
            # print(f'{col} has potential correlation with churn at {p}')
        corr_dict[col] = p
            
    
    # grabs 
    drop_extra = sorted(corr_dict, key=corr_dict.get)[degree:]
    
    df = df.drop(columns=drop_extra)
    
    return df
        

# Split given database
def split_df(df,strat_var,seed=123):
    """
    Returns three dataframes split from one for use in model training, validation, and testing. Takes two arguments:
        df: any dataframe to be split
        strat_var: the value to stratify on. This value should be a categorical variable.
    
    Function performs two splits, first to primarily make the training set, and the second to make the validate and test sets.
    """
    # Run first split
    train, validate_test = train_test_split(df,
                 train_size=0.60,
                random_state=seed,
                 stratify=df[strat_var]
                )
    
    # Run second split
    validate, test = train_test_split(validate_test,
                test_size=0.50,
                 random_state=seed,
                 stratify=validate_test[strat_var]
                )
    
    return train, validate, test
        

def drop_cols(df,cols=[],extras=False,degree=6):
    '''
    Drops columns. If no columns provided, then returns dataframe as is.
    
    Arguments:
    df: Required. DataFrame with columns to be dropped.
    cols: List, default is empty. If provided a list, then will drop the columns.
    extras: Default is False. If True, will run drop_extras function with provided degree.
        drop_extras will use a statistical test to determine a number of categorical columns to be dropped.
        Runs after other columns are dropped, which may impact the stats test run.
    degree: Default = 6. Used only in case extras is True.
    '''
    df = df.drop(columns=cols,errors='ignore')
    
    if extras == True:
        df = drop_extras(df,degree)
        
    return df
        

# create acquire function to prepare this nicely
def acquire_zillow():
    """
    Function to pull in zillow data. Returns a DataFrame. Also builds local csv file of the dataset.
    
    Parameters:
    -----------
    none
    
    """
    # build query
    query = """
select 
    bedroomcnt,
    bathroomcnt,
    calculatedfinishedsquarefeet,
    taxvaluedollarcnt,
    yearbuilt,
    taxamount,
    fips
from properties_2017
left join propertylandusetype
    using(propertylandusetypeid)
where propertylandusedesc = 'Single Family Residential'

    """
    
    # use env file to get database url
    url = env.get_db_url('zillow')
    filename = 'zillow.csv'
    
    df = w.check_file_exists(filename,query,url)
    
    return df


def prepare_zillow(df):
    """
    Cleans the zillow DataFrame
    
    Parameters:
    -----------
    df: DataFrame
        - The zillow DataFrame acquired from Codeup's MySQL database (or from local csv)
    
    """
    # re-assign as a copy to fix warnings
    df = df.copy()
    
    # drop the nulls
    df = df.dropna()
    
    # get a list of the cols and drop the two with decimals
    cols = list(df.columns)
    cols.remove('taxamount')
    cols.remove('bathroomcnt')
    cols.remove('propertylandusedesc')
    
    # convert columns of this list to integer
    for col in cols:
        df[col] = df[col].astype(int)
    
    return df


def split_continuous(df,seed=123):
    """
    Returns three dataframes split from one for use in model training, validation, and testing. 
    
    Function performs two splits, first to primarily make the training set, and the second to make the validate and test sets.
    
    Parameters:
    -----------
    df: DataFrame
        - the prepared dataset to be split
    seed: int, defaut=123
        - optional, a seed value to maintain consistency
    """
    # run first split
    train, validate_test = train_test_split(
        df,
        train_size = 0.6,
        random_state = 123
    )
    
    # run second split
    validate, test = train_test_split(
        df,
        train_size = 0.5,
        random_state = 123
    )
    
    return train,validate,test


def wrangle_zillow():
    """
    Acquires, prepares, and splits the zillow dataset.
    
    Parameters:
    -----------
    none
    
    
    """
    
    # acquire the data
    zillow = acquire_zillow()
    
    # prepare the data
    zillow = prepare_zillow(zillow)
    
    # split the data
    train,validate,test = split_continuous(zillow)
    
    return train,validate,test

