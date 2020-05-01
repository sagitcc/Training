import numpy as np
import pandas as pd
import time
import sklearn

@staticmethod
def timeit(method):
    def timed(*args, **kwargs):
        tstart = time.time()
        result = method(*args, **kwargs)
        tend = time.time()
        print('{0} : {1:2.2f} ms'.format(method.__name__, (tend - tstart) * 1000))
        return result
    return timed


def get_diff_return(data):

    diff_return = data['Close'].diff()    
    return diff_return

def get_log_return(data):

    log_return = data['Close'].apply(np.log).diff()
    return log_return


def get_relative_strength_index(data, window):
    
    def calculate(series):
        up = series[series>0].mean()
        down = -series[series<0].mean()
        return 100 * up/(up+down)

    diff_return = get_diff_return(data)
    relative_strength_index = diff_return.rolling(center=False, window=window).apply(calculate)

    return relative_strength_index


def get_bollinger_band(data, window, multiplier):

    middle_band = data['Close'].rolling(center=False, window=window).mean()
    sd = data['Close'].rolling(center=False, window=window).std()
    upper_band = middle_band + multiplier * sd
    lower_band = middle_band - multiplier * sd
    return upper_band, middle_band, lower_band


def get_aroon_oscillator(data, window):

    def calculate(series):
        period = len(series)
        new_series = series.reset_index(drop=True)
        idx_max = new_series.idxmax()+1
        idx_min = new_series.idxmin()+1
        return (idx_max-idx_min)/period*100

    aroon_oscillator = data['Close'].rolling(center=False, window=window).apply(calculate)
    return aroon_oscillator


def get_price_volume_trend(data):

    diff_return = get_diff_return(data)
    price_volume_trend = diff_return / data['Close'].shift(1) * data['Volume'].diff()
    
    return price_volume_trend


def get_acceleration_band(data, window):
    
    middle_band = data['Close'].rolling(center=False, window=window).mean()
    
    upper = data['High']*(1+2*(data['High']-data['Low'])/((data['High']+data['Low'])/2))
    lower = data['Low']*(1-2*(data['High']-data['Low'])/((data['High']+data['Low'])/2))
    
    upper_band = upper.rolling(center=False, window=window).mean()
    lower_band = lower.rolling(center=False, window=window).mean()

    return upper_band, middle_band, lower_band


def get_stochastic_oscillator(data, window):

    lowest_low = data['Low'].rolling(center=False, window=window).min()
    highest_high = data['High'].rolling(center=False, window=window).max()
    
    stochastic_oscillator_k = 100 * (data['Close'] - lowest_low) / (highest_high - lowest_low)
    stochastic_oscillator_d = stochastic_oscillator_k.rolling(center=False, window=3).mean()

    return stochastic_oscillator_k, stochastic_oscillator_d


def get_chaikin_money_flow(data, window):

    chaikin_money_flow = ((data['High'] + data['Low'] - 2 * data['Close']) * data['Volume']).rolling(center=False, window=window).sum() / data['Volume'].rolling(center=False, window=window).sum()

    return chaikin_money_flow

def get_parabolic_sar(data):
    pass


def get_price_rate_change(data, window):

    previous_value = data['Close'].shift(window)
    price_rate_change = (data['Close'] - previous_value) / previous_value

    return price_rate_change


def get_volume_weighted_average_price(data):
    
    volume_weighted_price = data['Volume'] * (data['High'] + data['Low']) / 2
    volume_weighted_average_price = volume_weighted_price.apply(np.cumsum) / data['Volume'].apply(np.cumsum)

    return volume_weighted_average_price


def get_momentum(data, window):
    
    momentum = data['Close'] - data['Close'].shift(window)

    return momentum


def get_commodity_channel_index(data, window, multiplier):

    average_price = (data['High'] + data['Low'] + data['Close']) / 3

    commodity_channel_index = (average_price - average_price.rolling(center=False, window=window).mean()) / average_price.rolling(center=False, window=window).std()

    return commodity_channel_index

def get_balance_volume(data):

    balance_volume = (data['Volume']*(2*~data['Close'].diff().le(0)-1)).cumsum()
    
    return balance_volume

def get_keltner_channels(data, window):

    upper = (4 * data['High'] - 2 * data['Low'] + data['Close']) / 3
    middle = (data['High'] + data['Low'] + data['Close']) / 3
    lower = (-2 * data['High'] + 4 * data['Low'] + data['Close']) / 3

    upper_band = upper.rolling(center=False, window=window).mean()
    middle_band = middle.rolling(center=False, window=window).mean()
    lower_band = lower.rolling(center=False, window=window).mean()

    return upper_band, middle_band, lower_band

def get_triple_exponential_moving_average(data, window):

    exponential_moving_average = data['Close'].ewm(span=window, min_periods=0, adjust=True, ignore_na=False).mean()

    triple_exponential_moving_average = 3 * exponential_moving_average - 3 * exponential_moving_average ** 2 + exponential_moving_average ** 3

    return triple_exponential_moving_average

def get_normalized_average_true_range(data, window):

    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift(1))
    low_close = abs(data['Low'] - data['Close'].shift(1))

    temp_table = pd.DataFrame({'hl':high_low, 'hc': high_close, 'lc':low_close})
    normalized_average_true_range = temp_table.max(axis=1).rolling(center=False, window=window).mean() / data['Close'] * 100

    return normalized_average_true_range


def get_average_directional_movement_index(data, window):

    up_move = data['High'] - data['High'].shift(1)
    down_move = data['Low'].shift(1) - data['Low'].shift(1)

    positive_move = pd.where((up_move > down_move) & (up_move > 0), up_move, 0)
    negative_move = pd.where((up_move < down_move) & (down_move > 0), down_move, 0)

    average_true_range = get_normalized_average_true_range(data, window)
    positive_index = 100 * (positive_move/average_true_range).ewm(span=window, min_periods=0, adjust=True, ignore_na=False).mean()
    negative_index = 100 * (negative_move/average_true_range).ewm(span=window, min_periods=0, adjust=True, ignore_na=False).mean()

    average_directional_movement_index = 100*(abs(positive_index-negative_index)/(positive_index+negative_index)).ewm(span=window, min_periods=0, adjust=True, ignore_na=False).mean()

    return average_directional_movement_index


def get_macd(data):

    ema12 = data['Close'].ewm(span=12, min_periods=0, adjust=True, ignore_na=False).mean()
    ema26 = data['Close'].ewm(span=26, min_periods=0, adjust=True, ignore_na=False).mean()

    return ema12-ema26


def get_money_flow_index(data, window):

    typical_price = (data['High'] + data['Low'] + data['Close'])/3
    positive_flow = pd.where(typical_price > typical_price.shift(1), typical_price, 0)
    negative_flow = pd.where(typical_price < typical_price.shift(1), typical_price, 0)
    money_flow_ratio = positive_flow.rolling(center=False, window=window).sum() / negative_flow.rolling(center=False, window=window).sum()
    money_flow_index = 100.0 - 100.0 / (1.0 + money_flow_ratio)

    return money_flow_index

def get_ichimoku_cloud(data):

    high9 = data['High'].rolling(center=False, window=9).max()
    low9 = data['Low'].rolling(center=False, window=9).min()
    high26 = data['High'].rolling(center=False, window=26).max()
    low26 = data['Low'].rolling(center=False, window=26).min()
    high52 = data['High'].rolling(center=False, window=52).max()
    low52 = data['Low'].rolling(center=False, window=52).min()

    turning_line = (high9 + low9)/2
    standard_line = (high26 + low26)/2

    ichimoku_span1 = ((turning_line + standard_line)/2).shift(26)
    ichimoku_span2 = ((high52 + low52)/2).shift(26)

    return ichimoku_span1, ichimoku_span2


def get_william_r(data, window):

    highest_high = data['High'].rolling(center=False, window=window).max()
    lowest_low = data['Low'].rolling(center=False, window=window).min()

    william_r = -100 * (highest_high - data['Close'])/(highest_high-lowest_low)

    return william_r


def get_adaptive_moving_average(data, window, power1, power2):

    abs_diff = (data['Close'] - data['Close'].shift(window)).abs()
    abs_diff_sum = (data['Close'] - data['Close'].shift(1)).rolling(window=window).sum()
    ratio = abs_diff / abs_diff_sum

    coefficient = (ratio*(2.0/(power1+1)-2.0/(power2+1))+2/(power2+1))**2
    adaptive_moving_average = data['Close']

    for i in range(len(1, adaptive_moving_average)):
        adaptive_moving_average.iloc[i] = adaptive_moving_average.iloc[i-1] + coefficient.iloc[i]*(data['Close']-adaptive_moving_average.iloc[i-1])
    
    return adaptive_moving_average
    