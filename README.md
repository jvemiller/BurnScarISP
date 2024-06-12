# BurnScarISP
Repository for ACENET ISP. Using machine learning to map the area of burn scars in Labrador.

The wildfire season broke a number of records in 2023 across Canada. This project looks to map the extent of wild fire damage, or burn scars, in Labrador. Three study areas were chosen based on a 2023 fire perimeter map (source: Canadian Wildland Fire Information System). Based on the data, there were 8 wildfires in Labrador between June and July. The three with the most damage done were chosen for analysis.

Imagery from the Sentinel-2 satellite was chosen for analysis. Sentinel-2 is a multispectral satellite with a spatial resolution ranging from 10 - 60 m, based on the spectral band. As healthy vegetation strongly reflects near infrared light, we're interested in near infrared bands like B8a. Satellite imagery of the study area from before and after the wildfire was collected. The bands in our data analysis will have a spatial resolution of 20 m. Data was downloaded from the Copernicus Data Space Ecosystem.

Different classification methods (supervised vs. unsupervised) and algorithms (random tree, support vector machines, etc.) will be tested and compared to see which gives the most accurate results. 

Location of wildfires:
Zone A - 58.9336521°W 52.0869104°N 
Zone B - 64.3448608°W 53.1020780°N 
Zone C - 62.0132970°W 56.9871025°N 
