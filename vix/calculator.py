import numpy as np
import pandas as pd
import datetime

class VixCalculator:

    def __init__(self, valuation_date, data_file):
        self.valuation_date = valuation_date
        self.data_file = data_file

    def get_near_next_date(self):
        valuation_day = self.valuation_date.weekday()
        if valuation_day < 2:
            near_delta = 3 * 7 + (4 - valuation_day)
        else:
            near_delta = 4 * 7 + (4 - valuation_day)
        next_delta = near_delta + 7
        near_date = self.valuation_date + datetime.timedelta(near_delta)
        next_date = self.valuation_date + datetime.timedelta(next_delta)
        if near_date.day >= 15 and near_date.day <= 21:
            near_date = datetime.datetime(near_date.year, near_date.month, near_date.day, 8, 30, 0)
        else:
            near_date = datetime.datetime(near_date.year, near_date.month, near_date.day, 15, 0, 0)
                    
        if next_date.day >= 15 and next_date.day <= 21:
            next_date = datetime.datetime(next_date.year, next_date.month, next_date.day, 8, 30, 0)
        else:
            next_date = datetime.datetime(next_date.year, next_date.month, next_date.day, 15, 0, 0)
        return near_date, next_date


    def read_data(self):
        date_parser = lambda x: datetime.datetime.strptime(x, '%m/%d/%Y')
        data = pd.read_csv(self.data_file, skiprows=[0,1], parse_dates=['Expiration Date'], date_parser=date_parser)
        return data


    def get_minute(self, term_date):
        
        term_delta = term_date - self.valuation_date
        minutes = term_delta.days * 24 * 60 + term_delta.seconds / 60
        return minutes

    def get_otm_idx(self, data, K0_idx, is_call=True):

        indices = list()
        if is_call:
            for i in range(K0_idx+1, data.shape[0]):
                if data['Bid'][i] != 0:
                    indices.append(i)
                elif i+1<data.shape[0] and data['Bid'][i+1] == 0:
                    break
        else:
            for i in range(K0_idx-1, -1, -1):
                if data['Bid.1'][i] != 0:
                    indices.append(i)
                elif i-1>=0 and data['Bid.1'][i-1] == 0:
                    break
        return indices


    def calculate_vol(self, data, term_date, rate, T):
        
        term_date_only = datetime.datetime(term_date.year, term_date.month, term_date.day)
        term_data = data[data['Expiration Date'] == term_date_only].reset_index()
        term_data['Calls_Avg'] = (term_data['Bid'] + term_data['Ask']) / 2
        term_data['Puts_Avg'] = (term_data['Bid.1'] + term_data['Ask.1']) / 2
        term_data['Diff'] = term_data['Calls_Avg'] - term_data['Puts_Avg']
        
        min_idx = term_data['Diff'].abs().idxmin()
        strike_level = term_data['Strike'][min_idx]
        F = strike_level + np.exp(rate * T) * term_data['Diff'][min_idx]
        K0_idx = term_data[term_data['Strike'] < F]['Strike'].idxmax()

        indices_call = self.get_otm_idx(term_data, K0_idx, True)
        indices_put = self.get_otm_idx(term_data, K0_idx, False)
        indices = indices_put + [K0_idx] + indices_call
        indices.sort()

        term_data['Mid_Quote'] = term_data['Calls_Avg']
        term_data.loc[indices_put, 'Mid_Quote'] = term_data['Puts_Avg'][indices_put]
        term_data.loc[K0_idx, 'Mid_Quote'] = (term_data['Calls_Avg'][K0_idx] + term_data['Puts_Avg'][K0_idx]) / 2

        vol_term1 = 0.0
        vol_term2 = 1.0 / T * (F / term_data['Strike'][K0_idx]-1)**2

        for i in range(len(indices)):
            if i == 0:
                delta_K = term_data['Strike'][indices[i+1]] - term_data['Strike'][indices[i]]
            elif i == len(indices)-1:
                delta_K = term_data['Strike'][indices[i]] - term_data['Strike'][indices[i-1]]
            else:
                delta_K = (term_data['Strike'][indices[i+1]] - term_data['Strike'][indices[i-1]]) / 2
            vol_term1 +=  delta_K / term_data['Strike'][indices[i]] ** 2 * term_data['Mid_Quote'][indices[i]]
            
        vol_term1 = 2.0 / T * np.exp(rate * T) * vol_term1
        vol = vol_term1 - vol_term2
        return vol



    def calculate_vix(self, rates):

        data = self.read_data()
        near_term_date, next_term_date = self.get_near_next_date()
        rate1, rate2 = rates[near_term_date], rates[next_term_date]
        
        minutes_month = 30 * 24 * 60
        minutes_year = 365 * 24 * 60
        N1, N2 = self.get_minute(near_term_date), self.get_minute(next_term_date)
        T1, T2 = N1/minutes_year, N2/minutes_year

        vol_near = self.calculate_vol(data, near_term_date, rate1, T1)
        vol_next = self.calculate_vol(data, next_term_date, rate2, T2)

        vix = 100 * np.sqrt((T1*vol_near*(N2-minutes_month)/(N2-N1) + T2*vol_next*(minutes_month-N1)/(N2-N1))*minutes_year/minutes_month)

        return vix
