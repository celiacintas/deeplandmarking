from src.dataload import DataLoader


def test_dataload():
    my_data_loader = DataLoader('data/test_image_landmarks.csv')
    X, y = my_data_loader.load_transform()
    assert (X.shape, y.shape) == ((2735, 9216), (2735, 90))


def test_network_build():
    pass


def test_training():
    pass


def test_pipeline():
    pass
