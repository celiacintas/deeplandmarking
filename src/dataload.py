import pandas as pd
import numpy as np
from os import path
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


class DataLoader(object):
    """
    Get images and landmarks from csv files
    @data_fn path to file with data
    @image_size int for square images (ej 96 == 96x96)
    @landmarks_n number of landmarks in the configuration
    @minmax retain the transformation for reconstruct coordinates
    """
    def __init__(self, data_fn):
        super(DataLoader, self).__init__()
        try:
            if path.exists(data_fn):
                self.data_fn = data_fn
            else:
                raise Exception
        except Exception as e:
            print("File Not Found")

    def load_transform(self, test=False):
        """
        Load images and landmarks coordinates and transform
        them as floats for gpu processing and [0, 1] range
        for images and [-1, 1] for landmarks coordinates.
        """
        df = pd.read_csv(self.data_fn)
        df['Image'] = df['Image'].apply(lambda im: np.fromstring(str(im), sep=' '))
        self.image_size = np.sqrt(df['Image'][0].shape[0])
        df = df.dropna()  # drop all rows that have missing values in them
        X = np.vstack(df['Image'].values) / 255.  # scale pixel to [0, 1]
        X = X.astype(np.float32)

        y = df[df.columns[:-1]].values
        self.landmarks_n = df.columns[:-1].shape[0]
        self.minmax = MinMaxScaler(feature_range=(-1, 1), copy=True)
        y = self.minmax.fit_transform(y)
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)

        return X, y
