import pandas as pd
import pytest

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def process_data(data):
    # 处理未标注的数据
    labeled_data = data[data['label'].notnull()]
    return labeled_data

def test_data_loading_and_processing():
    data = load_data('C://Users//Liu HaoTian//Desktop//rnn+tcn+transformer+kan//time_series//timeseries//examples//RLAD//clean_data.csv')
    assert data.shape == (6014, 115)
    
    processed_data = process_data(data)
    assert processed_data.shape[0] == 6014 - 111  # 处理未标注的111个数据
    assert processed_data.shape[1] == 115

pytest.main()