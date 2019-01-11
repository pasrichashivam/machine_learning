import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TimeSeriesAnomaly:
    def __init__(self, sigma, window_size, plot_data=True):
        self.sigma = sigma
        self.window_size = window_size
        self.plot_data = plot_data

    def get_moving_average(self, series):
        return series.rolling(self.window_size).mean()[self.window_size:]

    def get_moving_std(self, series):
        return series.rolling(self.window_size).std()[self.window_size:]

    # This returns the unusual datapoints in dataset with constant standard deviation throughout
    def get_unusual_points_std(self, data):
        _dict = dict()
        _list = list()
        mov_avg = self.get_moving_average(data)
        data = data[self.window_size:]
        residual = data - mov_avg
        std = np.std(residual)
        anomaly_index = [index for index, avg, orig in zip(data.index.values, mov_avg.values, data.values) if
                         orig > avg + std * self.sigma or orig < avg - std * self.sigma]
        anomalies = data[anomaly_index]
        if self.plot_data:
            plt.plot(data, "k.")
            plt.plot(mov_avg)
            plt.plot(anomalies, "r*", markersize=12)
            plt.show()
        for index, value in anomalies.iteritems():
            _dict['time_stamp'] = index.strftime("%Y-%m-%d %H:%M:%S")
            _dict['anomaly'] = value
            _list.append(_dict.copy())
        return _list

    # This returns the unusual datapoints in dataset with moving standard deviation.
    def get_unusual_points_rolling_std(self, data):
        _dict = dict()
        _list = list()
        mov_avg = self.get_moving_average(data)
        data = data[self.window_size:]
        residual = data - mov_avg
        rolling_std = self.get_moving_std(residual)
        anomaly_index = [index for index, avg, std, orig in
                         zip(data.index.values, mov_avg.values, rolling_std.values, data.values) if
                         orig > avg + std * self.sigma or orig < avg - std * self.sigma]
        anomalies = data[anomaly_index]
        for index, value in anomalies.iteritems():
            _dict['time_stamp'] = index.strftime("%Y-%m-%d %H:%M:%S")
            _dict['anomaly'] = value
            _list.append(_dict.copy())
        return _list

    def get_unusual_points(self, data, rolling_std=False):
        data.dropna(inplace=True)
        anomalies = self.get_unusual_points_rolling_std(data) if rolling_std else self.get_unusual_points_std(data)
        return anomalies


def read_csv_file():
    data = pd.read_csv("data/series_data.csv")
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d:%H:%M:%S")
    data.set_index(data['date'], inplace=True)
    data.sort_index(inplace=True)
    data = data['data']
    return data


if __name__ == "__main__":
    ts_anomalies = TimeSeriesAnomaly(sigma=2.8, window_size=20, plot_data=True)
    data = read_csv_file()
    data = data.resample("1H").mean().fillna(method="backfill")
    print(ts_anomalies.get_unusual_points(data))
