from utils import train_load, dev_load, test_load

TEST_PATH = 'data/test.preprocessed.npz'
TRAIN_PATH = 'data'
VAL_PATH = 'data/dev.preprocessed.npz'

test_trials, test_enrol, test_test = test_load(TEST_PATH)
print(test_trials)