import pandas as pd

def add_and_keep_lags_only(data, lags):
    """
    Add lags to a dataframe

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe to add lags to
    lags : int
        Number of lags to add

    Returns
    -------
    pd.DataFrame
        Dataframe with lags added
    """

    for c in data.columns:
        for lag in range(1, lags + 1):
            new_column = data[c].shift(lag)
            data = pd.concat([data, pd.DataFrame(new_column.values, columns=["{}(t-{})".format(c, lag)], index=data.index)], axis=1)
        
        data = data.drop(c, axis=1)
    
    return data

def str_2_bool(val):

    val = str(val)

    if val.lower() == "false":
        return False
    elif val.lower() == "true": 
        return True
    else:
        raise Exception("Invalid boolean value: {}".format(val))