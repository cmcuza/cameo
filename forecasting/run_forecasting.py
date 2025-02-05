from multiprocessing import Pool
from forecasting.forecasting_functions import process_data_item_forecasting

forecasting_model = 'lstm'

name_files_map = {
    # 'uci_electricity': (forecasting_model, 'electricity_hourly_dataset.tsf'),
    # 'aus_electricity': (forecasting_model, 'australian_electricity_demand_dataset.tsf'),
    # 'kdd_cup': (forecasting_model, 'kdd_cup_2018_dataset_without_missing_values.tsf'),
    # 'traffic': (forecasting_model, 'traffic_hourly_dataset.tsf'),
    'pedestrian': (forecasting_model, 'pedestrian_counts_dataset.tsf'),
    # 'rideshare': (forecasting_model, 'rideshare_dataset_without_missing_values.tsf')
}

if __name__ == "__main__":
    # with Pool() as pool:
    data_items = [(name, *paths) for name, paths in name_files_map.items()]
    process_data_item_forecasting(data_items[0])

    print('Done forecasting')