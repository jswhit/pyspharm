from matplotlib.toolkits.basemap import Basemap
from spharm import getgeodesicpts
import pylab as p
# set up orthographic map projection.
map = Basemap(projection='ortho',lat_0=50,lon_0=-100,resolution='l')
# draw coastlines, country boundaries, fill continents.
map.drawcoastlines()
map.fillcontinents(color='coral')
# draw the edge of the map projection region (the projection limb)
map.drawmapboundary()
# draw lat/lon grid lines every 30 degrees.
map.drawmeridians(p.arange(0,360,30))
map.drawparallels(p.arange(-90,90,30))
m = int(raw_input('input the number of points on the edge of each spherical geodesic triangle:'))
# find the geodesic points.
lats, lons = getgeodesicpts(m)
x, y = map(lons, lats)
map.scatter(x,y,marker='o',c='b',s=20,zorder=10)
p.show()

