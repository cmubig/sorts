from glob import glob
from os.path import join, abspath
from os import getcwd, stat
import os
import pandas as pd
import numpy
from numpy import arctan2, pi
from tqdm import tqdm
import pickle 

RUNWAY_ID = 'R2'
X_MIN = -0.2
X_MAX = 1.6
Y_MIN = -0.5
Y_MAX = 0.5

class CostMap:

    def __init__(self, path, data=None):
        self.path = path
        self.data = data
        self.names = ['x1', 'y1', 'z1', 'x2', 'y2', 'z2']
        self.columns = ['Frame#', 'AircraftID', 'x1', 'y1', 'z1', 'wind x (m/s)', 'wind y (m/s)']
        self.directions = ['W', 'E', 'NE', 'NW', 'N', 'SW', 'S', 'SE']
        self.keys = []
        self.state_direction = {}
        self.precision = 1

        if not self.load_dataframe():
            print("Generating cost map from",path)

            self.read_csv_folder()
            
            self.wind_direction_df()
            
            self.round_df()

            self.discretize_alt_df()

            self.calculate_angle_df()

            self.discretize_angle_df()

            self.set_keys()

            self.create_dictionary()

            self.clip_dictionary()

            self.normalize_dictionary()


            self.save_dataframe()

    def load_dataframe(self):
        path = self.path + '/file.pkl'
        if os.path.exists(path):
            print("Reading cost map from",path)
            with open(path,'rb') as f:
                self.state_direction = pickle.load(f)
            return True
        return False


    def read_csv_folder(self):
        df = pd.DataFrame(columns=self.names)
        dir_path = self.path
        full_path = join(abspath(getcwd()), dir_path, "*.txt")
        for file_name in tqdm(glob(full_path)):

            if stat(file_name).st_size == 0:
                continue

            csv_reader = pd.read_csv(file_name, header=None, delimiter=' ', names=self.columns)

            csv_reader = csv_reader[['x1', 'y1', 'z1', 'wind x (m/s)', 'wind y (m/s)']]
            csv_reader['x2'] = csv_reader['x1'].shift(-1)
            csv_reader['y2'] = csv_reader['y1'].shift(-1)
            csv_reader['z2'] = csv_reader['z1'].shift(-1)
            csv_reader.drop(csv_reader.tail(1).index, inplace=True)

            df = pd.concat([df, csv_reader], ignore_index=True)
        self.data = df

    def discretize_alt_df(self):
        self.data['z1'] = self.data['z1'].apply(self.discretize_altitude)
        self.data['z2'] = self.data['z2'].apply(self.discretize_altitude)

    def discretize_angle_df(self):
        self.data['direction'] = self.data['angle'].apply(self.discretize_angle)

    def calculate_angle_df(self):
        self.data['dx'] = self.data['x2'] - self.data['x1']
        self.data['dy'] = self.data['y2'] - self.data['y1']
        # print(self.data['dx'])
        # print(type(self.data['dx']))
        # print("\n")
        # print(self.data['dy'])
        # print(type(self.data['dy']))
        #self.data['angle'] = numpy.arctan2(self.data['dy'], self.data['dx']) * 180 / numpy.pi
        self.data['angle'] = self.data.apply(lambda df: numpy.arctan2(df['dy'], df['dx']) * 180 / numpy.pi, axis=1)

    def round_df(self):
        self.data = round(self.data, self.precision)

    def set_keys(self):
        self.data['right_direction'] = self.data['direction']
        self.keys = self.data.groupby(['x1', 'y1', 'z1', 'right_direction', 'wind_direction']).count()['direction'].to_dict().keys()

    def clip_dictionary(self, threshold=200):
        for key, value in self.state_direction.items():
            if value > threshold:
                self.state_direction[key] = threshold
            else:
                pass

    def create_dictionary(self):
        state_value = dict.fromkeys(self.keys, 0)
        for altitude in range(0, 6000, 1000):
            for wind_direction in [-1, 1]:
                for direction in self.directions:
                    dict_states = self.data[(self.data['z1'] == altitude) &
                                            (self.data['direction'] == direction) &
                                            (self.data['wind_direction'] == wind_direction)].groupby(
                        ['x1', 'y1', 'z1', 'right_direction', 'wind_direction']).count()['direction'].to_dict()
                    for key, value in dict_states.items():
                        if state_value[key] < value:
                            state_value[key] = value
        self.state_direction = state_value

    def save_dataframe(self):
        path = self.path + '/file.pkl'
        with open(path,'wb') as f:
            pickle.dump(self.state_direction,f)


    def state_value(self, x, y, z, angle, wind):

        x = round(x, 1)
        y = round(y, 1)
        z = self.discretize_altitude(z)
        angle = self.discretize_angle(angle)
        wind = wind
        if (x, y, z, angle, wind) in self.state_direction.keys():
            return self.state_direction[(x, y, z, angle, wind)]
        else:
            return 0.0

    def normalize_dictionary(self, normalization_range=(0, 1)):

        min_value = min(self.state_direction.values())
        max_value = max(self.state_direction.values())

        for key, value in self.state_direction.items():
            self.state_direction[key] = ((value - min_value) / (max_value - min_value)) * (
                        normalization_range[1] - normalization_range[0]) + normalization_range[0]

    @staticmethod
    def discretize_altitude(x):
        if x < 0.5 * 0.3048:
            return 0
        elif 0.5 * 0.3048 <= x < 1.5 * 0.3048:
            return 1000
        elif 1.5 * 0.3048 <= x < 2.5 * 0.3048:
            return 2000
        elif 2.5 * 0.3048 <= x < 3.5 * 0.3048:
            return 3000
        elif 3.5 * 0.3048 <= x < 4.5 * 0.3048:
            return 4000
        else:
            return 5000

    @staticmethod
    def discretize_angle(x):
        if -1 * 22.5 <= x < 1 * 22.5:
            return 'E'
        elif 1 * 22.5 <= x < 3 * 22.5:
            return 'NE'
        elif 3 * 22.5 <= x < 5 * 22.5:
            return 'N'
        elif 5 * 22.5 <= x < 7 * 22.5:
            return 'NW'
        elif -7 * 22.5 <= x < -5 * 22.5:
            return 'SW'
        elif -5 * 22.5 <= x < -3 * 22.5:
            return 'S'
        elif -3 * 22.5 <= x < -1 * 22.5:
            return 'SE'
        else:
            return 'W'
        
    @staticmethod
    def wind_direction(x):
        if x > 0:
            return 1
        else:
            return -1
        
    def wind_direction_df(self):
        self.data['wind_direction'] = self.data['wind x (m/s)'].apply(self.wind_direction)

if __name__ == '__main__':
    mp3d = CostMap('./dataset/111_days/processed_data/train')

   

    # input data sample
    x = 3.7 #(km)
    y = 2.7 #(km)
    z = 0.6096 #(km)
  
    angle = 11.5 #degrees

    print(f"({x},{y},{z},{angle}) =", mp3d.state_value(x, y, z, angle))
