import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from center_of_gravity import GravityCenters


def main():
    # Load network file
    cwd = os.getcwd()
    file_path = os.path.join(cwd, 'sample_data.csv')
    df = pd.read_csv(filepath_or_buffer=file_path,
                     delimiter=';',
                     dtype={'LocationName': str,
                            'Latitude': float,
                            'Longitude': float,
                            'Volume': int,
                            'LocationType': str})

    # Update inbound and outbound cost
    IB_OB_ratio = 2
    df['Cost'] = np.where(df.LocationType == 'Demand', df.Volume * IB_OB_ratio, df.Volume)

    # Get centers of gravity
    cogs = GravityCenters(lat=df.Latitude.values,
                          lng=df.Longitude.values,
                          weights=df.Cost.values,
                          method=KMeans,
                          n_clusters=2)
    cogs.fit_cogs()

    # Update network
    fit_cols = ['Latitude', 'Longitude']
    df = cogs.predict(df, df[fit_cols])
    print(df)
    cogs.plot(df)


if __name__ == '__main__':
    main()

