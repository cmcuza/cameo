## Datasets Used
The following datasets were used in this project. We acknowledge and give credit to the original sources:

1. **Household Electric Power Consumption ([HEPC](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption))**: DOI: 10.24432/C58K54
2. **Minimum Daily Temperatures in Melbourne ([MinTemp](https://www.kaggle.com/datasets/paulbrabban/daily-minimum-temperatures-in-melbourne))**
3. **Daily Pedestrian Counts ([Pedestrian](https://forecastingdata.org/))** 
4. **UK Historic Demand Data 2021 ([UKElecDEM](https://www.neso.energy/data-portal/historic-demand-data/historic_demand_data_2021))**
5. **Australia Electrical Demand ([AUSElecDem](https://forecastingdata.org/))** 
6. **Relative humidity ([Humidity](https://data.neonscience.org/data-products/DP1.00098.001/RELEASE-2023))**: DOI: 10.48443/g2j6-sr14 
7. **IR biological temperature ([IRBioTemp](https://data.neonscience.org/data-products/DP1.00005.001/RELEASE-2021))**: DOI:10.48443/jnwy-b177
8. **Seconds Solar Power ([SolarPower](https://forecastingdata.org/))**

## UCR Time Series Anomaly Detection datasets (2021)

You can download the UCR anomaly detection dataset from [figshare](https://figshare.com/articles/dataset/UCR_Time_Series_Anomaly_Detection_datasets_2021_/26410744?file=48036268):
```bash
# Download the dataset
wget https://figshare.com/ndownloader/files/48036268

# Unzip the dataset
unzip UCR_TimeSeriesAnomalyDatasets2021.zip -d UCR_TimeSeriesAnomalyDatasets2021

# Bring to the dataset to the current directory
mv -r UCR_TimeSeriesAnomalyDatasets2021/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData UCR_data
```

Once the data is there, you can run the anomaly detection experiments.


