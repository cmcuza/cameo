import os
import shutil
import pandas as pd
import gzip
import zipfile
import numpy as np
from os.path import join as path_join

class Loader:
    data: pd.DataFrame = None
    name = 'loader'
    harmonics = dict()
    seasonality = None
    aggregation = None

    def __init__(self, split=0.8):
        self.split = split
        self.data = pd.DataFrame()

    def load(self, file_path):
        pass

    def get_all_data(self):
        return self.data.values[:, 0]

    def get_train_test_values(self):
        b = int(len(self.data)*self.split)
        train = self.data.values[:b, 0]
        test = self.data.values[b:, 0]

        return train.copy(), test.copy()

    def get_train_test_df(self):
        b = int(len(self.data)*self.split)
        train = self.data.iloc[:b]
        test = self.data.iloc[b:]

        return train.copy(), test.copy()

    def prepare_for_parquet(self):
        return self.data

    def get_train_test_ts(self):
        b = int(len(self.data)*self.split)
        train = self.data.index[:b]
        test = self.data.index[b:]

        return train.copy(), test.copy()

    def rename_to_prophet(self):
        return self.data

    def rename_to_hwes(self):
        return self.data

    def increase_size(self, n):
        self.data = self.data.loc[self.data.index.repeat(n)]

    def to_gzip(self, root_src: str, root_dst: str):
        self.prepare_for_parquet()
        df = self.data.copy()
        df['datetime'] = None
        src = path_join(root_src, self.name + '.csv.gz')
        df.to_csv(src, compression='gzip')
        dst = path_join(root_dst, self.name + '.csv.gz')
        shutil.copy(src, dst)



class DataFactory:
    def __init__(self, split=0.8):
        self.split = split
        self.loaders = {
            'refit': RefitLoader,
            'hepc': HepcLoader,
            'min_temp': MinTemp,
            'pressure': PressureLoader,
            'aus_electrical_demand': AusElectricalDemand,
            'uk_electrical_demand': UkElectricalDemand,
            'wind4seconds': Wind4Seconds,
            'solar4seconds': Solar4Seconds,
            'pedestrian': PedestrianLoader,
            'biotemp': IRBioTemp,
            'humidity': HumidityLoader,
            'ucr': UCRLoader
        }

    def load_data(self, data_name, root_path):
        if data_name not in self.loaders:
            raise Exception(f'Unknown file: {data_name}')

        loader = self.loaders[data_name](self.split)
        loader.load(root_path)

        return loader

    def create_copy(self, data_loader: Loader, interp):
        loader = self.loaders[data_loader.name](self.split)
        loader.data = data_loader.data.copy()
        b = int(self.split * len(loader.data))
        if loader.data.shape[1] > 1:
            loader.data.loc[loader.data.index[:b], 'y'] = interp
        else:
            loader.data.iloc[:b] = interp[:, np.newaxis]

        return loader


class RefitLoader(Loader):
    file_name = 'refit_1.csv'
    name = 'refit'

    def load(self, file_path):
        self.data = pd.read_csv(path_join(file_path, self.file_name),
                                usecols=['Time', 'Aggregate'],
                                parse_dates=['Time'],
                                index_col=['Time']).sort_index()


class HepcLoader(Loader):
    file_name = 'hepc.csv'
    freq = '15T'
    fs = 4./3600.
    name = 'hepc'
    seasonality = 48

    def load(self, file_path):
        self.data = pd.read_csv(path_join(file_path, self.file_name), index_col='datetime', parse_dates=True).sort_index()

    def prepare_for_parquet(self):
        self.data.rename({'Global_active_power-R': 'y'}, axis=1, inplace=True)
        self.data.reset_index(inplace=True)


class MinTemp(Loader):
    file_name = 'daily-min-temperatures.csv'
    freq = 'D'
    fs = 1. / (24*3600)
    name = 'min_temp'
    seasonality = 365

    def __init__(self, split=0.7):
        super(MinTemp, self).__init__(0.7)
        self.split = 0.7

    def load(self, file_path):
        self.data = pd.read_csv(path_join(file_path, self.file_name), index_col='Date', parse_dates=True).sort_index()
        self.data.Temp = self.data.Temp.astype(float)
        date_range = pd.date_range(start=self.data.index.min(), end=self.data.index.max(), freq='D')
        self.data = self.data.reindex(date_range)
        self.data = self.data.interpolate()
        self.data[self.data.values == 0] = 0.01


class PedestrianLoader(Loader):
    file_name = 'pedestrian.csv'
    freq = 'H'
    fs = 1. / 3600
    name = 'pedestrian'
    seasonality = 24

    def load(self, file_path):
        self.data = pd.read_csv(path_join(file_path, self.file_name), index_col='ds', parse_dates=True).sort_index()
        self.data.y = self.data.y.astype(float)
        date_range = pd.date_range(start=self.data.index.min(), end=self.data.index.max(), freq='H')
        self.data = self.data.reindex(date_range)


class UkElectricalDemand(Loader):
    file_name = 'demanddata_2021.csv'
    freq = '30T'
    fs = 2. / 3600.
    name = 'uk_electrical_demand'
    seasonality = 48

    def load(self, file_path):
        self.data = pd.read_csv(path_join(file_path, self.file_name))
        ts = pd.date_range(start=self.data.SETTLEMENT_DATE.iloc[0],
                           periods=self.data.shape[0],
                           freq=self.freq, tz=None)
        self.data.set_index(ts, inplace=True)
        self.data = self.data[['ND']].astype(float)



class PressureLoader(Loader):
    file_name = 'Pressure.csv.gz'
    freq = '1T'
    name = 'pressure'

    def load(self, file_path):
        with gzip.open(path_join(file_path, self.file_name), 'rb') as f:
            self.data = pd.read_csv(f, header=None)
            ts = pd.date_range(start='08-2017', end='06-2021', periods=len(self.data))
            self.data.set_index(ts, inplace=True)
            self.data.drop([0], axis=1, inplace=True)
            self.data.rename({1: 'y'}, axis=1, inplace=True)


class MoteStrain(Loader):
    file_name = 'MoteStrain.csv.gz'
    freq = '1T'
    name = 'mote_strain'

    def load(self, file_path):
        with gzip.open(path_join(file_path, self.file_name), 'rb') as f:
            self.data = pd.read_csv(f, header=None)
            ts = pd.date_range(start='08-2017', end='06-2021', periods=len(self.data))
            self.data.set_index(ts, inplace=True)
            self.data.drop([0], axis=1, inplace=True)
            self.data.rename({1: 'y'}, axis=1, inplace=True)


class Wind4Seconds(Loader):
    file_name = 'wind_4_seconds_dataset.zip'
    freq = '4s'
    name = 'wind4seconds'
    aggregation = 450
    fs = 2. / 3600.

    def load(self, file_path):
        self.data = pd.DataFrame()

        with zipfile.ZipFile(path_join(file_path, self.file_name)) as _zip:
            for filename in _zip.namelist():
                with _zip.open(filename) as f:
                    for i, line in enumerate(f):
                        if i == 15:
                            line = str(line)
                            _, timestamp, points = line.split(':')
                            points = points.split(',')
                            points[-1] = points[-1][:-5]
                            self.data['y'] = np.asarray(points, dtype=float)
                            ts = pd.date_range(start=pd.to_datetime(timestamp), freq=self.freq, periods=len(self.data))
                            self.data.set_index(ts, inplace=True)


class Solar4Seconds(Loader):
    file_name = 'solar_4_seconds_dataset.zip'
    freq = '4s'
    name = 'solar4seconds'
    aggregation = 120
    seasonality = 24
    fs = 2. / 3600.

    def load(self, file_path):
        with zipfile.ZipFile(path_join(file_path, self.file_name)) as _zip:
            for filename in _zip.namelist():
                with _zip.open(filename) as f:
                    for i, line in enumerate(f):
                        if i == 15:
                            line = str(line)
                            _, timestamp, points = line.split(':')
                            points = points.split(',')
                            points[-1] = points[-1][:-5]
                            self.data['y'] = np.asarray(points, dtype=float)
                            ts = pd.date_range(start=pd.to_datetime(timestamp), freq=self.freq, periods=len(self.data))
                            self.data.set_index(ts, inplace=True)

        self.data = self.data.resample('30s').mean()
        print(self.data.shape)

    def prepare_for_parquet(self):
        self.data.reset_index(inplace=True)
        self.data.rename({'index': 'datetime'}, axis=1, inplace=True)


class AusElectricalDemand(Loader):
    file_name = 'australian_electricity_demand_dataset.zip'
    freq = '30T'
    fs = 2. / 3600.
    name = 'aus_electrical_demand'
    seasonality = 7
    aggregation = 48

    def load(self, file_path):
        with zipfile.ZipFile(path_join(file_path, self.file_name)) as _zip:
            for filename in _zip.namelist():
                with _zip.open(filename) as f:
                    for i, line in enumerate(f):
                        if i == 15:
                            line = str(line)
                            _, _, timestamp, points = line.split(':')
                            points = points.split(',')
                            points[-1] = points[-1][:-5]
                            self.data['y'] = np.asarray(points, dtype=float)
                            ts = pd.date_range(start=timestamp, freq=self.freq, periods=len(self.data), tz=None).tz_localize(None)
                            self.data.set_index(ts, inplace=True)

    def rename_to_prophet(self):
        # .dt.tz_localize(None)
        if 'ds' not in self.data.columns:
            self.data['cap'] = 13000
            self.data['floor'] = 3000
            self.data = self.data.reset_index().rename({'index': 'ds'}, axis=1)
        # self.data.ds = self.data.ds.tz_localize(None)
        return self.data

    def get_own_harmonics(self, K):
        harmonics = pd.DataFrame({'date': self.data.index})
        harmonics['date'] = pd.PeriodIndex(harmonics['date'], freq=self.freq)
        harmonics.set_index('date', inplace=True)
        harmonics.sort_index(inplace=True)
        for k in range(1, K+1):
            harmonics[f'sin-{k}har'] = np.sin(
                k * 2 * np.pi * (harmonics.index.hour * 60 + harmonics.index.minute) / (24 * 60))
            harmonics[f'cos-{k}har'] = np.cos(
                k * 2 * np.pi * (harmonics.index.hour * 60 + harmonics.index.minute) / (24 * 60))

        return harmonics

    def prepare_for_parquet(self):
        self.data.reset_index(inplace=True)
        self.data.rename({'index': 'datetime'}, axis=1, inplace=True)



class CricketLoader(Loader):
    file_name = 'Cricket.csv.gz'
    freq = '150s'
    fs = 2. / 3600.
    name = 'cricket'
    seasonality = 48

    def load(self, file_path):
        with gzip.open(path_join(file_path, self.file_name), 'r') as _gzip:
            self.data = pd.read_csv(_gzip, index_col=[0], names=['y'])


class WaferLoader(Loader):
    file_name = 'Wafer.csv.gz'
    freq = '150s'
    fs = 2. / 3600.
    name = 'wafer'
    seasonality = 48

    def load(self, file_path):
        with gzip.open(path_join(file_path, self.file_name), 'r') as _gzip:
            self.data = pd.read_csv(_gzip, index_col=[0], names=['y'])


class WindDirLoader(Loader):
    file_name = 'WindDirection.csv.gz'
    freq = '150s'
    fs = 2. / 3600.
    name = 'winddir'
    seasonality = 48

    def load(self, file_path):
        with gzip.open(path_join(file_path, self.file_name), 'r') as _gzip:
            self.data = pd.read_csv(_gzip, index_col=[0], names=['y'])


class WindSpeedLoader(Loader):
    file_name = 'WindSpeed.csv.gz'
    freq = '150s'
    fs = 2. / 3600.
    name = 'windspeed'
    seasonality = 48

    def load(self, file_path):
        with gzip.open(path_join(file_path, self.file_name), 'r') as _gzip:
            self.data = pd.read_csv(_gzip, index_col=[0], names=['y'])


class IRBioTemp(Loader):
    file_name = 'NEON_temp-bio'
    matching_pattern = 'NEON.D03.DSNY.DP1.00005.001.000.010.001.IRBT_1_minute'
    freq = '1T'
    fs = 1 / 60.
    name = 'biotemp'
    seasonality = 24
    aggregation = 60

    def load(self, file_path):
        self.data = pd.DataFrame()
        for root, dirs, files in os.walk(os.path.join(file_path, self.file_name)):
            for file in files:
                if file.find(self.matching_pattern) != -1:
                    df = pd.read_csv(os.path.join(root, file)).bioTempMean.to_frame()
                    if (df.bioTempMean.isnull().sum() != 0) or (df.bioTempMean.isna().sum() != 0):
                        continue

                    self.data = pd.concat([self.data, df])

        self.data.rename({'bioTempMean': 'y'}, axis=1, inplace=True)
        timestamp = '01/01/2016'
        ts = pd.date_range(start=pd.to_datetime(timestamp), freq=self.freq, periods=len(self.data))
        self.data.set_index(ts, inplace=True)

    def prepare_for_parquet(self):
        self.data.reset_index(inplace=True)
        self.data.rename({'index': 'datetime'}, axis=1, inplace=True)


class HumidityLoader(Loader):
    file_name = 'NEON_rel-humidity'
    matching_pattern = 'NEON.D03.DSNY.DP1.00098.001.000.040.001.RH_1min'
    freq = '1T'
    fs = 1 / 60.
    name = 'humidity'
    seasonality = 24
    aggregation = 60

    def load(self, file_path):
        self.data = pd.DataFrame()
        for root, dirs, files in os.walk(os.path.join(file_path, self.file_name)):
            for file in files:
                if file.find(self.matching_pattern) != -1:
                    df = pd.read_csv(os.path.join(root, file)).RHMean.to_frame()
                    if (df.RHMean.isnull().sum() != 0) or (df.RHMean.isna().sum() != 0):
                        if df.RHMean.isnull().sum() > 30:
                            continue
                        else:
                            df = df.interpolate(method='linear')

                    self.data = pd.concat([self.data, df])

        self.data.rename({'RHMean': 'y'}, axis=1, inplace=True)
        timestamp = '01/01/2016'
        ts = pd.date_range(start=pd.to_datetime(timestamp), freq=self.freq, periods=len(self.data))
        self.data.set_index(ts, inplace=True)

    def prepare_for_parquet(self):
        self.data.reset_index(inplace=True)
        self.data.rename({'index': 'datetime'}, axis=1, inplace=True)


class UCRLoader(Loader):
    file_name = 'UCR_anomaly_with_acf'

    def load(self, file_path):
        compiled_ucr_path = os.path.join('data', 'ucr.parquet')
        if os.path.exists(compiled_ucr_path):
            self.data = pd.read_parquet(compiled_ucr_path)
        else:
            ucr_path = os.path.join(file_path, self.file_name)
            loaded_data = []
            labels = []
            starts_disc = []
            ends_disc = []
            ends_training = []
            acfs = []
            for _, _, filenames in os.walk(ucr_path):
                for filename in filenames:
                    label = filename.split('_')
                    print(label)
                    acf = int(label[-4].split('.')[0])
                    end_training = int(label[-3])
                    start_disc = int(label[-2])
                    end_disc = int(label[-1].replace('.txt', ''))

                    full_path = os.path.join(ucr_path, filename)
                    elements = open(full_path, 'r').readlines()
                    if len(elements) < 2:
                        elements = elements[0].split('  ')

                    ts = [float(e) for e in elements[end_training:]]

                    loaded_data.append(np.array(ts))
                    labels.append(int(label[0]))
                    starts_disc.append(start_disc)
                    ends_disc.append(end_disc)
                    ends_training.append(end_training)
                    acfs.append(acf)

            new_tsf = pd.DataFrame({'label': labels,
                                    'series': loaded_data,
                                    'start_disc': starts_disc,
                                    'end_disc': ends_disc,
                                    'end_training': ends_training,
                                    'acf': acfs})

            self.data = new_tsf.sort_values('label').reset_index(drop=True)
            self.data.to_parquet(compiled_ucr_path)

