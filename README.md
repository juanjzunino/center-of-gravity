# Centers of Gravity

Python scripts to calculate centers of gravity using KMeans. Please refer to this medium post for more information: https://medium.com/analytics-vidhya/logistics-center-of-gravity-analysis-in-python-a21ad034f849

## Installation

`GravityCenters` can be installed cloning the repo from the command line:

`git clone https://github.com/juanjzunino/center-of-gravity.git`

Make sure to have all modules and their dependencies installed

`pip install -r requirements.txt`

## Quickstart

Import the `GravityCenters` class from `gravitycenters` and create an instance of the class `GravityCenters`

```python
from gravitycenters import GravityCenters

cogs = GravityCenters(lat=df.Latitude.values,
                      lng=df.Longitude.values,
                      weights=df.Cost.values,
                      method=KMeans,
                      n_clusters=n_clusters)
cogs.fit_cogs()
df = cogs.predict(df, df[fit_cols])
cogs.plot(df)
```
