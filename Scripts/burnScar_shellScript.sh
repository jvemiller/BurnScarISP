#!/bin/bash
#SBATCH --time=00:06:00
#SBATCH --account=an-tr043
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6500M
#SBATCH --output=burnScarOutput.txt

# add modules
module load python/3.11.5 scipy-stack

# set up virtual environment
virtualenv --no-download ENV
source ENV/bin/activate

# install python libraries
pip install --no-index --upgrade pip
pip install --no-index numpy matplotlib pandas fiona rasterio sklearn
pip install --no-index geopandas
pip install --no-index scikit-image

# run python scripts
python labradorBurnScarClassifier.py

srun hostname
