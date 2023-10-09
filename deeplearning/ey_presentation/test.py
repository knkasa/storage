import pandas as pd
import matplotlib.pyplot as plt
import pygeos as pg
import os
import numpy as np
from mpl_toolkits.basemap import Basemap

os.chdir('C:\my_working_env\deeplearning_practice\ey_presentation')


df_location = pd.read_csv('./data/olist_geolocation_dataset.csv')
df_location.drop( columns=['geolocation_zip_code_prefix'], axis=1, inplace=True )

latitude = df_location.geolocation_lat
longitude = df_location.geolocation_lng



# Create a scatter plot
plt.scatter(longitude, latitude, marker='o', color='red')

# Create a Basemap object
m = Basemap(llcrnrlon=min(longitude)-2, llcrnrlat=min(latitude)-2,
            urcrnrlon=max(longitude)+2, urcrnrlat=max(latitude)+2,
            )
            # resolution='l', projection='merc', lat_0=np.mean(latitude), lon_0=np.mean(longitude))

# Draw map features
m.drawcoastlines()
m.drawcountries()
#m.drawstates()

# Set plot limits
plt.xlim(min(longitude)-2, max(longitude)+2)
plt.ylim(min(latitude)-2, max(latitude)+2)

# Show the plot
plt.show()


