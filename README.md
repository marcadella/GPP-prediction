# GPP prediction from environmental time series

By Mayeul Marcadella

## Problem description

Advancement in environmental data collection and standardization has led to the establishment of datasets of long time series of environmental data sampling many biotopes worldwide, such as [FluxNet2015](https://fluxnet.org/data/fluxnet2015-dataset/).
Environmental variables from this dataset include variables measured locally with a half-hourly resolution such as temperature, precipitation, air pressure, and various variables related to gas fluxes.
Other datasets, such as [MODIS](https://modis.gsfc.nasa.gov/data/), propose variables collected through remote sensing technology. Examples are the Leaf Area Index (LAI) and the fraction of Absorbed Photosynthetically Active Radiation (fAPAR).

One interesting variable provided by FluxNet2015 is the Gross Primary Product (GPP) which is the brut carbon flux entering the ecosystem via photosynthesis.
In other words, it estimates the rate of capture of CO2 from the atmosphere by vegetation ecosystems (which is compensated by the re-emission of carbon from other processes such as respiration) and is therefore an important metric in the context of global warming and goals of achieving net-zero in 2050.

The questions I propose to examine are:
- How well is it possible to model GPP from local and/or remote sensed data using machine learning techniques?
- How well do models generalize through time and across vegetation types?

## Getting started

### Project structure

This project is composed of the following Jupyter notebooks:
- 0_background.ipynb
- 1_data_collection.ipynb
- 2_data_overview.ipynb
- 3_eda.ipynb
- 4_data_modeling.ipynb
- 5_model_comparison.ipynb

Each notebook saves the data necessary for later notebooks in the `resources` directory in such a way that the notebooks may be run in any order.
The raw data is also located in the `resources` directory.

Some functions are defined in python files included at the root of this repository.

### Requirements

Create a [new conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file):

```bash
conda env create -f environment.yml
```

Then activate the environment and start jupyter lab:
```bash
conda activate gpp
jupyter lab
```

## About this project

### Disclaimer

This project was done as an academic exercise and cannot be considered as a scientific contribution.
In particular, the author is not competent in the field of ecology.

### Acknowledgments

Many thanks to Pr. B. Stocker and the research group for Geocomputation and Earth Observation ([GECO](https://geco-group.org/)) at the Institute of Geography, University of Bern, for providing me with the project idea, data, and domain-specific knowledge required for carrying out this investigation.