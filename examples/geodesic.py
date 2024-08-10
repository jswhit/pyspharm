from mpl_toolkits.basemap import Basemap
from spharm import getgeodesicpts
import matplotlib.pyplot as plt
import numpy as np
# set up orthographic map projection.
map = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='l')
# draw coastlines, country boundaries, fill continents.
map.drawcoastlines()
map.fillcontinents(color='coral',lake_color='aqua')
# draw the edge of the map projection region (the projection limb)
map.drawmapboundary(fill_color='aqua')
# draw lat/lon grid lines every 30 degrees.
map.drawmeridians(np.arange(0,360,30))
map.drawparallels(np.arange(-90,90,30))
m = int(input('input the number of points on the edge of each spherical geodesic triangle:'))
# find the geodesic points.
lats, lons = getgeodesicpts(m)
x, y = map(lons, lats)
map.scatter(x,y,marker='o',c='b',s=20,zorder=10)
plt.show()
