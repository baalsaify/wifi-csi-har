import numpy as np

from csi_har.splits import cross_subject_folds


def test_no_subject_leakage_and_every_subject_tested_once():
    subjects = np.repeat(np.arange(1, 31), 20)  # 30 subjects x 20 trials
    tested = []
    for train, val, test in cross_subject_folds(subjects, n_folds=5, val_subjects=3, seed=0):
        s_train, s_val, s_test = (set(subjects[i]) for i in (train, val, test))
        assert s_train.isdisjoint(s_test)
        assert s_val.isdisjoint(s_test)
        assert s_train.isdisjoint(s_val)
        assert len(s_val) == 3
        assert len(train) + len(val) + len(test) == len(subjects)
        tested += sorted(s_test)
    assert sorted(tested) == list(range(1, 31))
