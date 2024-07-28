# BurnScarISP
Repository for ACENET ISP. Using machine learning to map the area of burn scars. The study area chosen is located in southern Labrador/Quebec. The study area was chosen based on a wildfire that occurred in 2023 at 58.9336521°W 52.0869104°N.

This project takes satellite data from the multi-spectral satellite Sentinel-2. By using our knowledge of the spectral reflectance of healthy vegetation and burnt areas, we can create a Normalized Burn Ratio (NBR) using near infrared and shortwave infrared bands. We can then perform machine learning on this NBR raster to classify which pixels are burnt and which are untouched.

This code performs 4 different classifications. The 4 classifications are unsupervised, Support Vector Machines (C-Support Vector), Random Forest, and Decision Trees. The code also creates a natural colour composite and a false colour composite of the study area.

The bands chosen for this analysis include the blue, green, and red bands, along with a near infrared band and a shortwave infrared band. A scene classification band was also used. The bands have a spatial resolution of 20m.

The study area was chosen based on data from the Canadian Wildland Fire Information System website. Historic data from this website was also used to help create training data for the supervised classifications.
