import torch
from sympy.codegen.ast import float64
from torch.utils.data import Dataset
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

class MyDataset(Dataset):
    def __init__(self,province):
        self.province = province
        self.raw_data = pd.read_excel('../data/final.xlsx')
        self.solar_data = pd.read_excel('../data/solar.xlsx')
        self.water_data = pd.read_excel('../data/water.xlsx')
        self.wind_data = pd.read_excel('../data/wind.xlsx')
        self.data = self.reduce_solar().to(dtype=torch.float64)

        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)
        self.data = torch.tensor(self.data).to(dtype=torch.float32)

    def reduce_solar(self):
        province_df = self.raw_data[self.raw_data['name']==self.province]
        solar_after_data = self.solar_data[self.solar_data['province']==self.province]
        solar_after_data['Year'] = solar_after_data['year'] % 100
        solar_after_data['Month'] = solar_after_data['month']
        solar_train = pd.merge(province_df,solar_after_data,how="right",on=['Year','Month'])
        solar_train = solar_train[['100 metre U wind component(m/s)',
                                   '100 metre V wind component(m/s)', '2 metre dewpoint temperature(K)',
                                   '2 metre temperature(K)', 'Surface runoff(m)', 'Runoff(m)',
                                   'Surface net short-wave (solar) radiation(J/m²)', '乘以面积(J)',
                                   'Surface short-wave (solar) radiation downward clear-sky(J/m²)',
                                   '乘以面积(J).1', 'Surface short-wave (solar) radiation downwards(J/m²)',
                                   '乘以面积(J).2','日发电系数']]
        solar_train = torch.tensor(solar_train.to_numpy())

        return solar_train

    def __getitem__(self, item):
        return self.data[item,:-1],self.data[item,-1]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = MyDataset("上海")
    print(dataset.data.shape)


