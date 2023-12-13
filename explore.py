
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats
import itertools
        

# Function to return whether answer on rejecting or failing to reject null hypothesis
def test_hypothesis(p, 
                    stat, 
                    tails='two', 
                    direction='greater',
                    alpha=0.05):
    '''
    test_hypothesis will take in a p value and a test statistic
    if p is less than a presumed alpha, then we  will reject
    our null hypothesis
    
    this takes in two positional arguments,
    p stat, a float value representing the probability of serendipity
    stat, a float value representing the test statistic
    
    with the keyword arguments f tails, direction, and alpha,
    the operator is able to change the control structrure in order
    to perform a one-tailed ttest if so desired
    '''
    if tails == 'two':
        if p < alpha:
            print(f'We reject the null hypothesis.\nOur p-value is {p} \nOur statistic value is {stat}')
        else:
            print(f'We fail to reject the null hypothesis.\nOur p-value is {p}.\nOur statistic is {stat}')
    else:
        if direction == 'greater':
            if ((p/2) < alpha) and (stat > 0):
                print(f'We reject our null hypothesis.\nOur p-value is {p} \nOur statistic value is {stat}')
            else:
                print(f'We fail to reject the null hypothesis.\nOur p-value is {p}.\nOur statistic is {stat}')
        else:
            if ((p/2) < alpha) and (stat < 0):
                print(f'We reject our null hypothesis.\nOur p-value is {p} \nOur statistic value is {stat}')
            else:
                print(f'We fail to reject the null hypothesis.\nOur p-value is {p}.\nOur statistic is {stat}')
        

def explore_univariate(train, cat_vars, quant_vars):
    for var in cat_vars:
        explore_univariate_categorical(train, var)
        print('_________________________________________________________________')
    for col in quant_vars:
        p, descriptive_stats = explore_univariate_quant(train, col)
        plt.show(p)
        print(descriptive_stats)
        

def explore_bivariate(train, target, cat_vars, quant_vars):
    for cat in cat_vars:
        explore_bivariate_categorical(train, target, cat)
    for quant in quant_vars:
        explore_bivariate_quant(train, target, quant)
        

def explore_multivariate(train, target, cat_vars, quant_vars):
    
    plot_swarm_grid_with_color(train, target, cat_vars, quant_vars)
    plt.show()
    violin = plot_violin_grid_with_color(train, target, cat_vars, quant_vars)
    plt.show()
    pair = sns.pairplot(data=train, vars=quant_vars, hue=target)
    plt.show()
    plot_all_continuous_vars(train, target, quant_vars)
    plt.show()
        

### Univariate
        

def explore_univariate_categorical(train, cat_var):
    """
    takes in a dataframe and a categorical variable and returns
    a frequency table and barplot of the frequencies. 
    """
    frequency_table = freq_table(train, cat_var)
    plt.figure(figsize=(2,2))
    sns.barplot(x=cat_var, y='Count', data=frequency_table, color='lightseagreen')
    plt.title(cat_var)
    plt.show()
    print(frequency_table)
        

def explore_univariate_quant(train, quant_var):
    """
    takes in a dataframe and a quantitative variable and returns
    descriptive stats table, histogram, and boxplot of the distributions. 
    """
    descriptive_stats = train[quant_var].describe()
    plt.figure(figsize=(8,2))

    p = plt.subplot(1, 2, 1)
    p = plt.hist(train[quant_var], color='lightseagreen')
    p = plt.title(quant_var)

    # second plot: box plot
    p = plt.subplot(1, 2, 2)
    p = plt.boxplot(train[quant_var])
    p = plt.title(quant_var)
    return p, descriptive_stats
        

def freq_table(train, cat_var):
    """
    for a given categorical variable, compute the frequency count and percent split
    and return a dataframe of those values along with the different classes. 
    """
    class_labels = list(train[cat_var].unique())

    frequency_table = (
        pd.DataFrame({cat_var: class_labels,
                      'Count': train[cat_var].value_counts(normalize=False), 
                      'Percent': round(train[cat_var].value_counts(normalize=True)*100,2)}
                    )
    )
    return frequency_table
        

#### Bivariate
        

def explore_bivariate_categorical(train, target, cat_var):
    '''
    takes in categorical variable and binary target variable, 
    returns a crosstab of frequencies
    runs a chi-square test for the proportions
    and creates a barplot, adding a horizontal line of the overall rate of the target. 
    '''
    print(cat_var, "\n_____________________\n")
    ct = pd.crosstab(train[cat_var], train[target], margins=True)
    chi2_summary, observed, expected = run_chi2(train, cat_var, target)
    p = plot_cat_by_target(train, target, cat_var)

    print(chi2_summary)
    print("\nobserved:\n", ct)
    print("\nexpected:\n", expected)
    plt.show(p)
    print("\n_____________________\n")
        

def explore_bivariate_quant(train, target, quant_var):
    '''
    descriptive stats by each target class. 
    compare means across 2 target groups 
    boxenplot of target x quant
    swarmplot of target x quant
    '''
    print(quant_var, "\n____________________\n")
    descriptive_stats = train.groupby(target)[quant_var].describe()
    average = train[quant_var].mean()
    mann_whitney = compare_means(train, target, quant_var)
    plt.figure(figsize=(4,4))
    boxen = plot_boxen(train, target, quant_var)
    swarm = plot_swarm(train, target, quant_var)
    plt.show()
    print(descriptive_stats, "\n")
    print("\nMann-Whitney Test:\n", mann_whitney)
    print("\n____________________\n")
        

## Bivariate Categorical
        

def run_chi2(train, cat_var, target):
    observed = pd.crosstab(train[cat_var], train[target])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    chi2_summary = pd.DataFrame({'chi2': [chi2], 'p-value': [p], 
                                 'degrees of freedom': [degf]})
    expected = pd.DataFrame(expected)
    return chi2_summary, observed, expected
        

def plot_cat_by_target(train, target, cat_var):
    p = plt.figure(figsize=(2,2))
    p = sns.barplot(cat_var, target, data=train, alpha=.8, color='lightseagreen')
    overall_rate = train[target].mean()
    p = plt.axhline(overall_rate, ls='--', color='gray')
    return p
        

## Bivariate Quant
        

def plot_swarm(train, target, quant_var):
    average = train[quant_var].mean()
    p = sns.swarmplot(data=train, x=target, y=quant_var, color='lightgray')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p
        

def plot_boxen(train, target, quant_var):
    average = train[quant_var].mean()
    p = sns.boxenplot(data=train, x=target, y=quant_var, color='lightseagreen')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p
        

# alt_hyp = ‘two-sided’, ‘less’, ‘greater’
        

def compare_means(train, target, quant_var, alt_hyp='two-sided'):
    x = train[train[target]==0][quant_var]
    y = train[train[target]==1][quant_var]
    return stats.mannwhitneyu(x, y, use_continuity=True, alternative=alt_hyp)
        

### Multivariate
        

def plot_all_continuous_vars(train, target, quant_vars):
    '''
    Melt the dataset to "long-form" representation
    boxenplot of measurement x value with color representing the target variable. 
    '''
    my_vars = [item for sublist in [quant_vars, [target]] for item in sublist]
    sns.set(style="whitegrid", palette="muted")
    melt = train[my_vars].melt(id_vars=target, var_name="measurement")
    plt.figure(figsize=(8,6))
    p = sns.boxenplot(x="measurement", y="value", hue=target, data=melt)
    p.set(yscale="log", xlabel='')    
    plt.show()
        

def plot_violin_grid_with_color(train, target, cat_vars, quant_vars):
    cols = len(cat_vars)
    for quant in quant_vars:
        _, ax = plt.subplots(nrows=1, ncols=cols, figsize=(16, 4), sharey=True)
        for i, cat in enumerate(cat_vars):
            sns.violinplot(x=cat, y=quant, data=train, split=True, 
                           ax=ax[i], hue=target, palette="Set2")
            ax[i].set_xlabel('')
            ax[i].set_ylabel(quant)
            ax[i].set_title(cat)
        plt.show()
        

def plot_swarm_grid_with_color(train, target, cat_vars, quant_vars):
    cols = len(cat_vars)
    for quant in quant_vars:
        _, ax = plt.subplots(nrows=1, ncols=cols, figsize=(16, 4), sharey=True)
        for i, cat in enumerate(cat_vars):
            sns.swarmplot(x=cat, y=quant, data=train, ax=ax[i], hue=target, palette="Set2")
            ax[i].set_xlabel('')
            ax[i].set_ylabel(quant)
            ax[i].set_title(cat)
        plt.show()
        

def plot_variable_pairs(df):
    '''
    Takes the zillow DataFrame and plots all the pairwise relationships.
    
    Parameters:
    -----------
    df: DataFrame
        - the train DataFrame to be plotted
    
    Returns:
    --------
    None
    
    
    '''
    
    # get all the numerical columns
    numeric_columns = [col for col in df.columns if df[col].dtype != 'O']

    # use itertools to get combinations
    col_combos = list(itertools.combinations(numeric_columns,2))
    
    # plot each pairwise relationship in a for loop
    plt.figure(figsize=(16,20))
    for i,cols in enumerate(col_combos):
        plot_number = i+1
        plt.subplot(7,3,plot_number)
        plt.title(f'{cols[0]} paired with {cols[1]}')
        sns.regplot(y=df[cols[0]],x=df[cols[1]],line_kws={'color':'orange'})
        plt.xlabel(cols[1])
        plt.ylabel(cols[0])
        plt.grid(alpha=0.4)
    plt.tight_layout()

    plt.show()
    return None


def plot_categorical_and_continuous_vars(df,cat_col,con_col):
    '''
    Plots a sample of the dataset with a barplot, strip plot, and boxenplot.
    
    Parameters:
    -----------
    df: DataFrame
        - A DataFrame to be plotted
    cat_col: str
        - The categorical column of the DataFrame to be plotted
    con_col: str
        - The continuous column of the DataFrame to be plotted
    sample_size: int, default=10**3
        - The size of the samples to be plotted
    
    
    '''
    
    plt.figure(figsize=(16,8))
    
    # plot first figure
    plt.subplot(3,1,1)
    sns.barplot(data=df,x=cat_col,y=con_col)
    # plot second figure
    plt.subplot(3,1,2)
    sns.stripplot(data=df,x=cat_col,y=con_col)
    # plot third figure
    plt.subplot(3,1,3)
    sns.boxenplot(data=df,x=cat_col,y=con_col)

    plt.tight_layout()
    plt.show()
    return None

