import numpy as np
import pandas as pd
import pickle

id_column = '0_ID'
time_column = '1_TIME'
feature_columns = ['BID_P_1', 'BID_P_2', 'BID_P_3', 'BID_P_4', 'BID_P_5', 'BID_P_6', 'BID_P_7', 'BID_P_8', 'BID_P_9', 'BID_P_10', 'BID_V_1', 'BID_V_2', 'BID_V_3', 'BID_V_4', 'BID_V_5', 'BID_V_6', 'BID_V_7', 'BID_V_8', 'BID_V_9', 'BID_V_10',
                   'ASK_P_1', 'ASK_P_2', 'ASK_P_3', 'ASK_P_4', 'ASK_P_5', 'ASK_P_6', 'ASK_P_7', 'ASK_P_8', 'ASK_P_9', 'ASK_P_10', 'ASK_V_1', 'ASK_V_2', 'ASK_V_3', 'ASK_V_4', 'ASK_V_5', 'ASK_V_6', 'ASK_V_7', 'ASK_V_8', 'ASK_V_9', 'ASK_V_10']


class Model:

    def __init__(self, model_file, data_file):
        self.model = pickle.load(open(model_file, 'rb'))
        self.model.verbose = False
        self.model.early_stopping = False
        self.meta = pd.read_hdf(data_file, 'meta')

        self.mean_feature_values = pd.read_hdf(data_file, 'mean_feature_values')
        self.std_feature_values = pd.read_hdf(data_file, 'std_feature_values')

        self.mean_target_value = self.meta['mean_target_value'][0]
        self.std_target_value = self.meta['std_target_value'][0]

        self.mean_time_diff = self.meta['mean_time_diff'][0]
        self.std_time_diff = self.meta['std_time_diff'][0]

        self.min_prices = []

        self.last_time = None
        self.time_diffs = []
        self.model.best_loss_ = 1.0
        self.Xs = []


    def injest_line(self, csv_line):
        X = (csv_line[feature_columns] - self.mean_feature_values) /  self.std_feature_values

        time = pd.to_datetime(csv_line[time_column]).iloc[0]

        if self.last_time:
            time_diff = (time - self.last_time).microseconds
            time_diff = (time_diff - self.mean_time_diff) / self.std_time_diff
        else:
            time_diff = 0.0

        self.last_time = time
        X = np.concatenate([[[time_diff]], X], axis=1)

        self.min_prices.append((csv_line['BID_P_1'].iloc[0] + csv_line['ASK_P_1'].iloc[0])/2)
        self.min_prices = self.min_prices[-100:]

        self.Xs.append(X)
        self.Xs = self.Xs[-100:]
        self.X = X

    def y_to_Y(self, y):
        Y = y * self.std_target_value + self.mean_target_value
        return Y

    def Y_to_y(self, Y):
        Y = (Y - self.mean_target_value) / self.std_target_value
        return Y

    def predict(self):
        y_pred = self.model.predict(self.X)

        Y_true = np.std(self.min_prices)

        if len(self.Xs) >= 100:
            self.model.partial_fit(self.Xs[0], [self.Y_to_y(Y_true)])

        return self.y_to_Y(y_pred)[0]
