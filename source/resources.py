import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from statsmodels.regression import linear_model
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import itertools as it
from IPython.display import Markdown, display
import time

def stepwise_selection(X, y,
                       initial_list=[],
                       threshold_in=0.01,
                       threshold_out = 0.05,
                       verbose=False):
    """ Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """

    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print(f'Add {best_feature} with p-value {best_pval}')

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print(f'Drop {worst_feature} with p-value {worst_pval}')
        if not changed:
            break
    return included

def bin_data(data,  bins, cats):
    '''
    Takes a series, a list of of bin edge definitions, and a list of category names.
    Transforms the series to categorical values, then does dummy encoding.
    Returns a datatrame of the dummy variables.
    '''
    bins = pd.IntervalIndex.from_tuples(bins)
    bds_bins = pd.cut(data, bins)
    bds_bins.categories = cats
    return pd.get_dummies(bds_bins.cat.rename_categories(bds_bins.categories), drop_first=True)

def preprocess_data(data, cats=[], cols_to_drop=[], trans=False, t_list=[], scaled=True):
    '''
    Arguments:
        data - Dataframe to preprocess
        cats - List of category names when bining
        cols_to_drop - List of column names to drop
        trans - Boolean indicating whether to transform the data.
        t_list - List of features to transform. If empty and trans=True, will transform all numerical values.
        scaled - Boolean that if true, will scale down the values with sqft to units of 100sqft
    '''

    # Make a copy so that the original DataFrame is not altered.
    data_wc = data.copy()

    # Build a list of sub-DataFrames for later concatenation
    to_concat = []

    # Iterate over category list and encode. Original column will appended to cols_to_drop so that all encoded columns can
    # later be dropped from our data_wc
    # The DataFrames holding the dummie variables will be added to to_concat and concatednated back to the original dataframe
    # We use Pandas 'cut' function to bin the data for 'bedrooms' and 'bathrooms'. Refer to bin_data() function.
    for cat in cats:
        # Use binning to create categories, then dummy encode.
        if cat == 'bedrooms':
            bed_bins = [(0, 2), (2, 3), (3,4), (4, 5), (5, 11)]
            bed_cats = ['less than two bedroom','2 bedroom','3 bedroom','4 bedroom','5 or more bedroom']
            to_concat.append(bin_data(data_wc[cat], bed_bins, bed_cats))
            cols_to_drop.append(cat)

        # Use binning to create categories, then dummy encode.
        elif cat == 'bathrooms':
            bath_bins =[(0.0, 1.0), (1.0, 1.75), (1.75,2.5), (2.5, 3.0), (3.0, 3.75), (3.75,8)]
            bath_cats = ['less than one bath','1 - 1.75 bath','1.75 - 2.5 bath','2.5 - 3.0 bath','3.0 - 3.75 bath', '3.75 bath and up']
            bathroom_dummies = bin_data(data_wc[cat], bath_bins, bath_cats)
            to_concat.append(bathroom_dummies)
            cols_to_drop.append(cat)

        # Use label encoding
        elif cat == 'waterfront':
            waterf_df = pd.DataFrame()
            waterf_df[cat] = data_wc.apply(lambda row: set_waterfront(row), axis=1)
            waterf_df[cat] = waterf_df[cat].astype('category')
            waterf_df[cat] = waterf_df[cat].cat.codes
            to_concat.append(waterf_df)
            cols_to_drop.append(cat)

        # Use dummy encoding
        elif cat == 'condition':
            data_wc[cat] = data_wc[cat].astype('category')
            to_concat.append(pd.get_dummies(data_wc[cat], prefix='condition_', drop_first=True))
            cols_to_drop.append(cat)

        # Use dummy encoding
        elif cat == 'zipcode':
            to_concat.append(pd.get_dummies(data_wc[cat], prefix='z_', drop_first=True))
            cols_to_drop.append(cat)

        # Use dummy encoding
        elif cat == 'floors':
            to_concat.append(pd.get_dummies(data_wc[cat], prefix='floors_', drop_first=True))
            cols_to_drop.append(cat)

        # Use dummy encoding
        elif cat == 'yr_renovated':
            data_wc[cat] = data_wc[cat].apply(lambda x: 1 if x > 0 else -1 if np.isnan(x) else x).astype('category')
            data_wc[cat].categories = ['unknow','not_renovated','renovated']
            to_concat.append(pd.get_dummies(data_wc[cat].cat.rename_categories(data_wc[cat].categories), drop_first=True))
            cols_to_drop.append(cat)

    # This dataframe will be returned if the target was included, else it will remain empty
    df_y = pd.DataFrame()
    if 'price' in data_wc.columns:
        df_y = data_wc[['price']]
        # scaled to units of $1000
        df_y = df_y / 1000
        cols_to_drop.append('price')

    # Verify that columns in cols_to_list are valid. if cols_to_drop has invalid entries, let's catch them here.
    cols_to_drop = [c for c in cols_to_drop if c in data_wc.columns]
    data_wc.drop(cols_to_drop, axis=1, inplace=True)

    # Fix entry in 'sqft_basement'. Only if this feature wasn't dropped
    if 'sqft_basement' in data_wc.columns:
        data_wc['sqft_basement'].replace('?','0.0',inplace=True)
        data_wc['sqft_basement'] = data_wc['sqft_basement'].astype(str).astype(float).astype(int)

    # Remove '?' in 'waterfront' if not passed as categorical and it wasn't dropped
    if 'waterfront' in data_wc.columns:
        data_wc['waterfront'] = data_wc.apply(lambda row: set_waterfront(row), axis=1)

    # Create new feature - 'age_when_sold' - only if ther other two features are both in dataframe.
    if 'date' in data_wc.columns:
        if 'yr_built' in data_wc.columns:
            data_wc['age_when_sold'] = pd.to_datetime(data_wc['date']) - pd.to_datetime(data['yr_built'])
            data_wc['age_when_sold'] = data_wc['age_when_sold'].dt.days
            data_wc['date'] = data_wc['date'].str.replace("/","").astype(float)
            data_wc.drop(['date','yr_built'],axis=1)
        else:
            # If 'yr_built' was dropped, but not 'data'
            data_wc['date'] = data_wc['date'].str.replace("/","").astype(float)

    # scale to units of 100 sqft
    if scaled:
        sqft = ['sqft_living','sqft_lot','sqft_above','sqft_lot15','sqft_living15']
        for c in data_wc:
            if c in sqft:
                data_wc[c] = data[c] / 100

    # Transform the numerical columns
    new_df = pd.DataFrame()
    if trans:
        # We log transform and scale numerical data.
        if not t_list:
            d_l = [y for y in data_wc.columns if y in ['long','lat']]
            t_list = data_wc.drop(d_l,axis=1).columns
        else:
            new_df = data_wc.copy()
        for c in t_list:
            logc = np.log(data_wc[c]+1)
            new_df[c] = (logc-min(logc))/(max(logc)-min(logc))


    # Drop the individual indexes before concatenating the dataframes. This will prevent index alignment issues.
    for tc in to_concat:
        tc.reset_index(drop=True, inplace=True)

    # If we transformed the data, we use the dataframe created, else just use same DataFrame
    # We then concatenate all DataFrames
    new_df = data_wc if new_df.empty else new_df
    new_df.reset_index(drop=True, inplace=True)
    df_y.reset_index(drop=True, inplace=True)
    to_concat = [new_df] + to_concat + [df_y]

    # Concatenate all of the dataframes to produce the final set
    X = pd.concat(to_concat, axis=1)

    # Set the target to 0 if this is the test set, else return the target dataframe
    y = 0 if df_y.empty else df_y

    return X,y

def exp_interactions(X,y,n=5):
    # Takes dataframe of predictors and the series containing targets.
    # Returns a list of tuples containing the top_n interaction pairs plus their R2 scores.

    # Instantiate our model
    regression = LinearRegression()

    # Get the baseline 'r2' score
    crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
    baseline = np.mean(cross_val_score(regression, X, y, scoring='r2', cv=crossvalidation))

    # Create list to hold interaction tuples
    interactions = []

    # Create a list of all 2 feature combinations
    feat_combinations = it.combinations(X.columns, 2)

    data = X.copy()
    # Iterate over the list of feature pairs, creating the interaction term and adding it to the model.
    # Only one interaction term is used each time so that we can compare 'r2' impact.
    for i, (a, b) in enumerate(feat_combinations):
        data['interaction'] = data[a] * data[b]
        score = np.mean(cross_val_score(regression, data, y, scoring='r2', cv=crossvalidation))
        if score > baseline:
            interactions.append((a, b, round(score,3)))

        if i % 500 == 0:
            print(' '*12,end='\r')
            print(i,end='\r')

    # Capture the top_n scores in a list.
    top_n_int = sorted(interactions, key=lambda inter: inter[2], reverse=True)[:n]
    #print(f'Top {n} interactions: \n{top_n_int}')
    return top_n_int

def add_interactions(df, interactions):
    '''
    Takes the training set and a list of interaction tuples to create a dataframe of interactions.
    Returns the the training set concatenated with the interactions
    '''
    # Create the interactions
    interact_df = pd.DataFrame()
    for inter in interactions:
        interact_df[f'{inter[0]}:{inter[1]}'] = df[inter[0]] * df[inter[1]]

    # Concatenate with the training set
    df.reset_index(drop=True, inplace=True)
    interact_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, interact_df],axis=1)

    return df

# https://answiz.com/questions/35235/find-p-value-significance-in-scikit-learn-linearregression
def regression_analysis(X,predictions,params):
    newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
    MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))

    # Note if you don't want to use a DataFrame replace the two lines above with
    # newX = np.append(np.ones((len(X),1)), X, axis=1)
    # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b

    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

    sd_b = np.round(sd_b,3)
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilites"] = [params,sd_b,ts_b,p_values]
    print(myDF3)

def set_waterfront(s):
    '''
    Rough classifier based on where on the map waterfront properties appear.
    Takes a dataframe row and uses the 'lat' and 'long' fields to determine if they are within either of three bounding
    boxes that contain most of the waterfront properties.
    '''
    if pd.notna(s['waterfront']):
        return s['waterfront']

    elif (47.34 < s['lat'] < 47.6) and (-122.6 < s['long'] < -122.35):
        return 1.0
    elif (47.5 < s['lat'] < 47.78) and (-122.27 < s['long'] < -122.2):
        return 1.0
    elif (47.56 < s['lat'] < 47.66) and (-122.1 < s['long'] < -122.05):
        return 1.0
    else:
        return 0.0


def md(string):
    display(Markdown(string))

def plot_trans(predictor, target, z=0):
    '''
    This function takes two pandas series - predictor and target, as well as an additional argument 'z'.
    It produces a grid of six plots. The first row contains three distribution plots - one of the predictor
    untransformed, another of the log-transormed predictor, and the third of the square root transformed predictor.

    The second row contains three scatter plots, again of the predictor against the target unchanged, and transformed.
    The optional argument 'z' is used for the log transformation where there are zero values.
    '''
    # Get the transforms
    tran_log_lot = np.log(predictor + z)
    tran_sqrt_lot = np.sqrt(predictor)

    fig,ax = plt.subplots(2,3,figsize=(16,8))
    plt.subplots_adjust(wspace = 0.8)
    trans_lst = [predictor, tran_log_lot, tran_sqrt_lot]
    titles = ['Unchanged','Log transformed','SQRT transformed']
    for i,t in enumerate(trans_lst*2):
        if i > 2:
            sns.regplot(x=t,y=target, ax=ax[i//3][i%3])
        else:
            sns.distplot(t, ax=ax[i//3][i%3])
            ax[i//3][i%3].set_title(titles[i])
