from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


def oversample(X_train, y_train):
    rus = RandomUnderSampler()
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

    return X_train_rus, y_train_rus


def undersample(X_train, y_train):
    ros = RandomOverSampler()
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

    return X_train_ros, y_train_ros

