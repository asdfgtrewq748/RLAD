import pytest

@pytest.fixture(scope='session')
def setup_data():
    # 在这里加载和处理数据
    pass

@pytest.fixture(scope='function')
def setup_model():
    # 在这里设置模型
    pass