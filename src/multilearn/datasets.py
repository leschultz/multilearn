import pandas as pd


def load(locations, names=None, targets=None, drops=None):
    '''
    Load data included with the package.

    Args:
        locations (list): The loctions of data to load.
        names (list): The names for each dataset.
        targets (list): The name of the target variable for each of names.
        drops (list): A list of list for columns to drop.

    Returns:
        Dict: A dictionary with features, targets, and names.
    '''

    namescond = names is None
    dropscond = drops is not None

    data = {}
    for count in range(len(locations)):

        if namescond:
            name = count
        else:
            name = names[count]

        if dropscond:
            drop = drops[count]

        target = targets[count]

        df = locations[count]
        df = pd.read_csv(df)

        if drop is None:
            X = df.drop(target, axis=1).values
        else:
            X = df.drop([target]+drop, axis=1).values

        y = df[target].values

        data[name] = {}
        data[name]['X'] = X
        data[name]['y'] = y

    return data
