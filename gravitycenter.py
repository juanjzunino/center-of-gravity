import warnings
import random
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

warnings.filterwarnings('ignore')
plt.style.use('ggplot')


class GravityCenters:
    """
    Calculate Centers of Gravity for a set of X,Y points
    using KMeans algorithm

    Attributes
    ----------
    lat: List[float]
        Latitude
    lng: List[float]
        Longitude
    weights: Optional[List[float]
        Weights for fitting the model
    method: KMeans | MiniBatchKMeans
        Clustering algorithm
    n_clusters: int
        Number of gravity centers
    kmeans: Kmeans | MiniBatchKMeans
        Fitted instance of the clustering algorithm
    cogs_df: pd.DataFrame
        Center of Gravity Data.Frame
    cogs_latitude: List[float]
        Center's of gravity latitude
    cogs_longitude: List[float]
        Center's of gravity longitude
    """

    def _init_(self,
               lat: List[float],
               lng: List[float],
               weights: Optional[List[float]] = None,
               method=KMeans,
               n_clusters: int = 2) -> None:
        self.lat = lat
        self.lng = lng
        self.weights = weights
        self.method = method
        self.n_clusters = n_clusters
        self.kmeans = None
        self.cogs_df = None
        self.cogs_latitude = None
        self.cogs_longitude = None

    def fit_cogs(self) -> None:
        """
        Fits clustering model and generate a new DataFrame with the result
        """
        # Initialize cluster method
        self.kmeans = self.method(init='random',
                                  n_clusters=self.n_clusters,
                                  random_state=0)

        # Fit model
        self.kmeans.fit(X=np.column_stack((self.lat, self.lng)),
                        sample_weight=self.weights)

        # Update coordinates
        cogs = self.kmeans.cluster_centers_
        self.cogs_latitude = cogs[:, 0]
        self.cogs_longitude = cogs[:, 1]

        # Generate df
        self.cogs_df = self._generate_df()

    def _generate_df(self) -> pd.DataFrame:
        """
        Generates DataFrame based on models fitting
        """
        return (pd.DataFrame({"CoGLatitude": self.cogs_latitude, "CoGLongitude": self.cogs_longitude})
                  .reset_index()
                  .rename(columns={"index": "ID"}))

    def get_latitude(self) -> List[float]:
        """
        Return CoG Latitude
        """
        return self.cogs_latitude

    def get_longitude(self) -> List[float]:
        """
        Return CoG Longitude
        """
        return self.cogs_longitude

    def predict(self, df: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
        """
        Assign points to a CoG

        Parameters
        ----------
        df: pd.DataFrame
            Network DataFrame
        x: pd.DataFrame | array
            Features used to predict

        Returns
        -------
        df: pd.DataFrame
            Updated DataFrame with predictions
        """
        df["Cluster"] = self.kmeans.predict(x)
        df = pd.merge(df,
                      self.cogs_df,
                      left_on="Cluster",
                      right_on="ID",
                      how="inner")
        return df

    def plot(self, df) -> None:
        """
        Plots network with COG
        """
        cogs_color = {cog: (random.random(), random.random(), random.random()) for cog in df.Cluster.unique()}

        for i, row in df.iterrows():
            x = [row.Longitude, row.CoGLongitude]
            y = [row.Latitude, row.CoGLatitude]
            plt.plot(x, y, marker="o", color=cogs_color[row.Cluster])
        plt.show()
