# Wild Fire Burn Scar Classifier
# John Miller
# ACENET Certificate in Advanced Computing
# labradorBurnScarClassifier.py

# This code uses machine learning to classify burn scars from wildfires. 
# Our study area is located in southern Labrador/Quebec. 
# Sentinel-2 satellite data will be used for our analysis.
# Satellite data is dated September 22nd, 2023.


# import libraries
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona

import matplotlib.pyplot as plt
from matplotlib import colors, cm

import rasterio as rio
from rasterio.plot import show

from skimage import exposure, morphology
from sklearn import cluster
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)


# Function to create some simple plots
def create_plot(img, title, out_fig):
    
    # Define a plot
    fig, ax = plt.subplots(figsize=(7,7), dpi=200)

    ax.set_xlabel("Eastings (Metres)")
    ax.set_ylabel("Northings (Metres)")
    plt.yticks(rotation='vertical')
    show(img, transform=transform, cmap=cmap, title=title, ax=ax)

    # Save fig
    plt.savefig(out_fig, bbox_inches='tight')

    # Close fig
    plt.close()
    

# Function to create confusion matrices and do predictions
def predict_matrix(classification, clf, out_fig):
    
    # Create class list
    classes = ['Untouched', 'Burn Scar']
    
    # Fit the model
    clf.fit(X_train, y_train)

    clf_pred = clf.predict(X_test)

    # Get labels for confusion matrix
    labels = clf.classes_

    # Calculate accuracy
    accuracy = accuracy_score(y_test,clf_pred)
    print(f"{classification} Accuracy: {accuracy}")

    # Calculate precision
    precision = precision_score(y_test, clf_pred, average = 'macro')
    print(f"{classification} Precision: {precision}")

    # Calculate recall 
    recall = recall_score(y_test, clf_pred, average = 'macro')
    print(f"{classification} Recall: {recall}")

    # Create confusion Matrix
    conf_matrix = confusion_matrix(y_test, clf_pred, labels=labels)

    # Plot confusion matrix
    cmd = ConfusionMatrixDisplay(conf_matrix, display_labels = classes)
    cmd.plot(cmap='Greens')
    
    title = classification + " Confusion Matrix"

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.yticks(rotation='vertical')

    # Save fig
    plt.savefig(out_fig, dpi=200, bbox_inches='tight')

    # Close fig
    plt.close()

    # Let's perform the classification on our data

    classification_pred = clf.predict(nbr_1d).reshape(empty_array[:,:,0].shape)
    
    return classification_pred


# Function to clean the noise from our classifications
def clean_noise(classification, raster_path, out_path, out_fig):
    
    # Read in the classified raster and convert to boolean
    raster = rio.open(raster_path)
    array = raster.read().astype(bool)

    # Use skimage to remove pixel clusters that are smaller in size
    cleaned = morphology.remove_small_objects(
        array, min_size=300, connectivity=1)

    # Get count of True pixels (pixels that are burned)
    burn_pixels = np.count_nonzero(cleaned == True)

    # Get total burnt area in hectares
    # Number of pixels times 20x20 m pixel size divided by 10000
    total_area_hect = burn_pixels * 400 / 10000
    # How many American football fields is that?
    football = (total_area_hect * 2.471) / 1.32

    print(f"Based on our {classification}, as of Sept 22, 2023, an estimated "
          f"{total_area_hect:,.2f} hectares of land has been damaged by "
          "wildfires within our study area.")
    print(f"That's the equivalent of {football:,.2f} American football "
          "fields!\n")

    with rio.open(out_path, 'w', **meta) as dest:
        dest.write(cleaned.astype(rio.float32))

    # Let's view the cleaned classification
    # Define a plot
    fig, ax = plt.subplots(figsize=(7,7), dpi=200)
    
    title = "Burn Scar Locations (" + classification + ")"
    ax.set_xlabel("Eastings (Metres)")
    ax.set_ylabel("Northings (Metres)")
    plt.yticks(rotation='vertical')
    hect_str = str(total_area_hect) + " Ha Burnt"
    ax.text(775000, 5695000, hect_str, fontsize=11, 
            bbox={'facecolor': 'white'})
    show(cleaned, transform=transform, cmap='Reds', title=title, ax=ax)

    # Save fig
    plt.savefig(out_fig, bbox_inches='tight')

    # Close fig
    plt.close()
    
    return cleaned, hect_str


    
# Open bands used in analysis
# blue band
b2 = rio.open("../Raster/T20UQC_20230922T150719_B02_20m.jp2")
# green band
b3 = rio.open("../Raster/T20UQC_20230922T150719_B03_20m.jp2")
# red band
b4 = rio.open("../Raster/T20UQC_20230922T150719_B04_20m.jp2")
# near infrared band
b8a = rio.open("../Raster/T20UQC_20230922T150719_B8A_20m.jp2")
# short wave infrared band
b12 = rio.open("../Raster/T20UQC_20230922T150719_B12_20m.jp2")
# scene classification
scl = rio.open("../Raster/T20UQC_20230922T150719_SCL_20m.jp2")

# Set path to training data shapefile
training_path = "../Vector/trainingData_pts.shp"

# Set out path for training data with ID
train_ID_path = "../Vector/trainingData_withID.shp"

# Set Raster out paths
# Natural Colour Composite Raster
out_raster_nat = "../Raster/studyArea_naturalColour.tif"
# False Colour Composite Raster
out_raster_false = "../Raster/studyArea_falseColour.tif"
# Normalized Burn Ratio Raster
out_raster_nbr = "../Raster/studyArea_NBR.tif"
# Water Mask Raster
water_path = '../Raster/studyArea_waterMask.tif'
# Uncleaned Unsupervised Classification Raster
unsupervised_path = '../Raster/unsupervisedClassification_uncleaned.tif'
# Cleaned Unsupervised Classification Raster
unsup_burnScar_path = '../Raster/burnScars_unsupervisedClassification.tif'
# Uncleaned SVM Classification Raster
svm_path = '../Raster/svmClassification_uncleaned.tif'
# Cleaned SVM Classification Raster
svm_burnScar_path = '../Raster/burnScars_svmClassification.tif'
# Uncleaned Random Forest Classification Raster
rf_path = '../Raster/rfClassification_uncleaned.tif'
# Cleaned Random Forest Classification Raster
rf_burnScar_path = '../Raster/burnScars_rfClassification.tif'
# Uncleaned Decision Tree Classification Raster
dt_path = '../Raster/dtClassification_uncleaned.tif'
# Cleaned Decision Tree Classification Raster
dt_burnScar_path = '../Raster/burnScars_dtClassification.tif'

# Set classification strings for names & titles
unsup_class = "Unsupervised Classification"
svm_class = "SVM Classification"
rf_class = "Random Forest Classification"
dt_class = "Decision Tree Classification"


# Let's make a natural colour composite
# Take the metadata from one of our bands
meta = b2.meta
# Update the meta veriable so that count = 3 (3 bands)
meta.update({"count":3})

# Write the red, green, and blue bands to the out raster
# **meta adds the metadata we saved
with rio.open(out_raster_nat, 'w', **meta) as dest:
    dest.write(b4.read(1),1)
    dest.write(b3.read(1),2)
    dest.write(b2.read(1),3)

# Open natural composite raster
natural_comp_raster = rio.open(out_raster_nat)

# Read natural composite raster as an array
natural_comp_array = natural_comp_raster.read()

# Transpose the array
natural_comp_array = natural_comp_array.transpose(1,2,0)

# Clip the lower and upper 2 percentiles to elminate a potential contrast 
# decrease caused by outliers
p2, p98 = np.percentile(natural_comp_array, (2,98)) 

# Use skimage to rescale the intensity. The values will range from 0 to 1
natural_comp_array = exposure.rescale_intensity(natural_comp_array, 
                                                in_range=(p2, p98)) / 100000

# Create transformed variable for our plots (makes x & y Eastings & Northings)
transform = natural_comp_raster.transform

# Set variables for plot
cmap = None
title = "Natural Colour Composite of Study Area"
out_fig = "../Figures/naturalColour.png"

# Call create_plot function
create_plot(natural_comp_array.transpose(2,0,1), title, out_fig)


# Let's make a false colour composite
# Write the shortwave infrared, infrared, and blue bands to the out raster
with rio.open(out_raster_false, 'w', **meta) as dest:
    dest.write(b12.read(1),1)
    dest.write(b8a.read(1),2)
    dest.write(b2.read(1),3)


# Open false composite raster
false_comp_raster = rio.open(out_raster_false)

# Read false composite raster as an array
false_comp_array = false_comp_raster.read()

# Transpose the array
false_comp_array = false_comp_array.transpose(1,2,0)

# Clip the lower and upper 2 percentiles to elminate a potential contrast 
# decrease caused by outliers
p2, p98 = np.percentile(false_comp_array, (2,98)) 

# Use skimage to rescale the intensity. The values will range from 0 to 1
false_comp_array = exposure.rescale_intensity(false_comp_array, 
                                              in_range=(p2, p98)) / 100000

# Set variables for plot
cmap = None
title = "False Colour Composite of Study Area"
out_fig = "../Figures/falseColour.png"

# Call create_plot function
create_plot(false_comp_array.transpose(2,0,1), title, out_fig)


# Create a Normalized Burn Ratio (NBR)
# NBR = (NIR - SWIR) / (NIR + SWIR)
# Read in NIR & SWIR bands
nir = b8a.read()
swir = b12.read()

# Calculate NBR
nbr = (nir.astype(float)-swir.astype(float))/(nir+swir)

# Extract metadata from one of the bands and update
# This meta variable will be used throughout code
meta = b12.meta
meta.update(driver='GTiff')
meta.update(dtype=rio.float32)

# Write the NBR to the out raster
with rio.open(out_raster_nbr, 'w', **meta) as dest:
    dest.write(nbr.astype(rio.float32))


# Let's view the Normalized Burn Ratio raster
nbr_raster = rio.open(out_raster_nbr)
nbr_array = nbr_raster.read()

# Define a plot
fig, ax = plt.subplots(figsize=(8,8), dpi=200)

# Display NBR
title = "Normalized Burn Ratio of Study Area"
ax.set_xlabel("Eastings (Metres)")
ax.set_ylabel("Northings (Metres)")
plt.yticks(rotation='vertical')
show(nbr_array, transform=transform, cmap='gist_gray', title=title, ax=ax)
fig.colorbar(cm.ScalarMappable
             (norm=colors.Normalize(vmin=np.nanmin(nbr_array), 
                                    vmax=np.nanmax(nbr_array)), 
              cmap='gist_gray'), ax=ax)

# Save fig
out_fig = "../Figures/normalizedBurnRatio.png"
plt.savefig(out_fig, bbox_inches='tight')

# Close fig
plt.close()


# Let's create a water mask
# The scl raster is a classified raster. Pixels with a value of 6 are water
# Read in the scl raster
scl_array = scl.read()

# Create a water mask by making all values != 6 equal to 0
# Converting to 0 will help when adding the water mask to the 
# unsupervised classification
water_mask = np.where(scl_array != 6, 0, scl_array)

with rio.open(water_path, 'w', **meta) as dest:
    dest.write(water_mask.astype(rio.float32))

# Set variables for plot
cmap = 'Blues'
title = "Water Mask of Study Area"
out_fig = "../Figures/waterMask.png"

# Call create_plot function
create_plot(water_mask, title, out_fig)


# Perform an unsupervised classification
# Create an empty array with the same dimensions and data type as our 
# raster array
empty_array = np.empty((nbr_raster.height, nbr_raster.width, nbr_raster.count), 
                       nbr_raster.meta['dtype'])

# Loop through the NBR's bands to fill the empty array
for band in range(empty_array.shape[2]):
    empty_array[:,:,band] = nbr_raster.read(band+1)

# Convert to 1d array
nbr_1d = empty_array[:,:,:3].reshape(
    (empty_array.shape[0]*empty_array.shape[1],empty_array.shape[2]))

# Perform K Means cluster on NBR 
cl = cluster.KMeans(n_clusters=2)
param = cl.fit(nbr_1d)

# Get the labels of the classes and reshape back to original rows & cols 
# (5490, 5490)
img_cl = cl.labels_
img_cl = img_cl.reshape(empty_array[:,:,0].shape)

# Set variables for plot
cmap = 'Reds'
title = "Unsupervised Classification (No Water Mask)"
out_fig = "../Figures/unsupervised_noMask.png"

# Call create_plot function
create_plot(img_cl, title, out_fig)

# Remove water from classified raster
water_raster = rio.open(water_path)
water_array = water_raster.read()

no_water = img_cl.astype(rio.float32) + water_array.astype(rio.float32)

classified_no_water = np.where(no_water > 1, 0, no_water)

with rio.open(unsupervised_path, 'w', **meta) as dest:
    dest.write(classified_no_water.astype(rio.float32))

# Let's view the unsupervised classification with the water mask applied
# Set variables for plot
cmap = 'Reds'
title = "Unsupervised Classification (Water Mask Applied)"
out_fig = "../Figures/unsupervised_waterMask.png"

# Call create_plot function
create_plot(classified_no_water, title, out_fig)

# Clean the unsupervised classification by removing noise
out_fig = "../Figures/unsupervised_burnScars.png"
unsup_cleaned, unsup_hect = clean_noise(unsup_class, unsupervised_path, 
                                        unsup_burnScar_path, out_fig)


# Let's train our data
# Read in training data shapefile using Geopandas
training_data = gpd.read_file(training_path)

# Display training data
training_data.plot(column='Classname', cmap=None, legend=True, markersize=12, 
                   figsize=(7, 7))

plt.title("Trained Data")
plt.xlabel("Eastings (Metres)")
plt.ylabel("Northings (Metres)")
plt.yticks(rotation='vertical')

# Save fig
out_fig = "../Figures/trainingData.png"
plt.savefig(out_fig, dpi=200, bbox_inches='tight')

# Close fig
plt.close()

# Add ID column to training data
training_data = training_data.assign(id=range(len(training_data)))

# Save a copy of the training data with the ID column
training_data.to_file(train_ID_path)

# Read training data as DataFrame
training_df = pd.DataFrame(training_data.drop(
    columns=['geometry', 'Classcode']))
# Create an empty series
sampled = pd.Series()

# Read input shapefile with fiona and iterate over each feature
with fiona.open(train_ID_path) as shp:
    for feature in shp:
        siteID = feature['properties']['id']
        coords = feature['geometry']['coordinates']
        # Read pixel value of NBR raster at the given coordinates and add to 
        # series
        value = [v for v in nbr_raster.sample([coords])]
        sampled.loc[siteID] = value

# Reshape sampled values and add to new DataFrame
df = pd.DataFrame(sampled.values.tolist(), index=sampled.index)
df['id'] = df.index
df = pd.DataFrame(df[0].values.tolist(), columns=['PixelValue'])
df['id'] = df.index

# Merge sampled DataFrame and training Dataframe
data = pd.merge(df, training_df, on ='id')

# Set X & y
X = data.iloc[:,0:1].values
y = data.iloc[:,3].values

# Train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, 
                                                    stratify = y)


# Let's perform a Supervised Classification using Support Vector Machines 
# (C-Support Vector Classification)

# Let's fit our training data, create a confusion matrix, & perform a 
# SVM classification
clf = SVC(kernel='rbf')
out_fig = "../Figures/svm_confMatrix.png"
svm_pred = predict_matrix(svm_class, clf, out_fig)

# Set variables for plot
cmap = 'Reds'
title = "SVM Classification (No Water Mask)"
out_fig = "../Figures/svm_noMask.png"

# Call create_plot function
create_plot(svm_pred, title, out_fig)

# Remove water from classified raster
svm_no_water = svm_pred.astype(rio.float32) + water_array.astype(rio.float32)
svm_no_water = np.where(svm_no_water > 1, 0, svm_no_water)

with rio.open(svm_path, 'w', **meta) as dest:
    dest.write(svm_no_water.astype(rio.float32))

# Set variables for plot
cmap = 'Reds'
title = "SVM Classification (Water Mask Applied)"
out_fig = "../Figures/svm_waterMask.png"

# Call create_plot function
create_plot(svm_no_water, title, out_fig)

# Clean the svm classification by removing noise
out_fig = "../Figures/svm_burnScars.png"
svm_cleaned, svm_hect = clean_noise(svm_class, svm_path, 
                                    svm_burnScar_path, out_fig)


# Let's perform a Supervised Classification using Random Forest Classifier

# Let's fit our training data, create a confusion matrix, & perform a 
# Random Forest classification
clf = RandomForestClassifier(random_state=42)
out_fig = "../Figures/rf_confMatrix.png"
rf_pred = predict_matrix(rf_class, clf, out_fig)

# Set variables for plot
cmap = 'Reds'
title = "Random Forest Classification (No Water Mask)"
out_fig = "../Figures/rf_noMask.png"

# Call create_plot function
create_plot(rf_pred, title, out_fig)

# Remove water from classified raster
rf_no_water = rf_pred.astype(rio.float32) + water_array.astype(rio.float32)
rf_no_water = np.where(rf_no_water > 1, 0, rf_no_water)

with rio.open(rf_path, 'w', **meta) as dest:
    dest.write(rf_no_water.astype(rio.float32))

# Set variables for plot
cmap = 'Reds'
title = "Random Forest Classification (Water Mask Applied)"
out_fig = "../Figures/rf_waterMask.png"

# Call create_plot function
create_plot(rf_no_water, title, out_fig)

# Clean the rf classification by removing noise
out_fig = "../Figures/rf_burnScars.png"
rf_cleaned, rf_hect = clean_noise(rf_class, rf_path, rf_burnScar_path, out_fig)


# Supervised Classification using Decision Trees

# Let's fit our training data, create a confusion matrix, & perform a 
# Decision Tree classification
clf = DecisionTreeClassifier(random_state=42)
out_fig = "../Figures/dt_confMatrix.png"
dt_pred = predict_matrix(dt_class, clf, out_fig)

# Set variables for plot
cmap = 'Reds'
title = "Decision Tree Classification (No Water Mask)"
out_fig = "../Figures/dt_noMask.png"

# Call create_plot function
create_plot(dt_pred, title, out_fig)

# Remove water from classified raster
dt_no_water =dt_pred.astype(rio.float32) + water_array.astype(rio.float32)
dt_no_water = np.where(dt_no_water > 1, 0, dt_no_water)

with rio.open(dt_path, 'w', **meta) as dest:
    dest.write(dt_no_water.astype(rio.float32))

# Set variables for plot
cmap = 'Reds'
title = "Decision Tree Classification (Water Mask Applied)"
out_fig = "../Figures/dt_waterMask.png"

# Call create_plot function
create_plot(dt_no_water, title, out_fig)

# Clean the dt classification by removing noise
out_fig = "../Figures/dt_burnScars.png"
dt_cleaned, dt_hect = clean_noise(dt_class, dt_path, dt_burnScar_path, out_fig)


# Let's add all our classifications to one figure
fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(nrows=2, ncols=2, 
                                             figsize=(10,10), dpi=200)

fig.suptitle('Burn Scar Locations by Classification', y=0.95, fontsize=16)
fig.supxlabel("Eastings (Metres)", y=0.05)
fig.supylabel("Northings (Metres)", x=0.05)

title1 = "Unsupervised Clustering"
title2 = "SVM Classification"
title3 = "Random Forest Classification"
title4 = "Decision Tree Classification"

ax1.tick_params(axis='y', labelrotation=90)
ax2.tick_params(axis='y', labelrotation=90)
ax3.tick_params(axis='y', labelrotation=90)
ax4.tick_params(axis='y', labelrotation=90)

ax1.text(769500, 5695000, unsup_hect, fontsize=9, bbox={'facecolor': 'white'})
ax2.text(769500, 5695000, svm_hect, fontsize=9, bbox={'facecolor': 'white'})
ax3.text(769500, 5695000, rf_hect, fontsize=9, bbox={'facecolor': 'white'})
ax4.text(769500, 5695000, dt_hect, fontsize=9, bbox={'facecolor': 'white'})

# Display classifications
show(unsup_cleaned, transform=transform, cmap='Reds', title=title1, ax=ax1)
show(svm_cleaned, transform=transform, cmap='Reds', title=title2, ax=ax2)
show(rf_cleaned, transform=transform, cmap='Reds', title=title3, ax=ax3)
show(dt_cleaned, transform=transform, cmap='Reds', title=title4, ax=ax4)

# Save fig
out_fig = "../Figures/burnScars_allClassifications.png"
plt.savefig(out_fig, bbox_inches='tight')

# Close fig
plt.close()