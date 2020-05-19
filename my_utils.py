


############################################################
### I. Utils ###############################################

### Evaluate #####################
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix

def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_scores = model.predict_proba(X)[:, 1]

    print('Accuracy: ', accuracy_score(y, y_pred))
    print('AUC: ', roc_auc_score(y, y_scores))
    print('AP: ', average_precision_score(y, y_scores))
    print('\nConfusion Matrix')
    print(confusion_matrix(y, y_pred))
    return

### Combine Features #############
from scipy.sparse import csr_matrix, coo_matrix, hstack

def combine_features(feature_groups):
    '''
      Input: list of features. eg. [user_features, rating_features, review_features]
    '''
    sparse_features = [csr_matrix(f) for f in feature_groups]
    return hstack( sparse_features, format='csr' )

############################################################
############################################################



############################################################
### II. Imbalance ##########################################

### 1. Class Weight ##############
from sklearn.utils.class_weight import compute_class_weight

def class_weight(y):
    return compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y)

### 2. Oversample ################
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

def random_oversample(X, y):
    rus = RandomUnderSampler()
    X_rus, y_rus = rus.fit_resample(X, y)
    return X_rus, y_rus

def smote(X, y):
    X_smote, y_smote = SMOTE().fit_resample(X, y)
    return X_smote, y_smote

def adasyn(X, y):
    X_ADASYN, y_ADASYN = ADASYN().fit_resample(X, y)
    return X_ADASYN, y_ADASYN

### 3. Undersample ################
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss

def ranodm_undersample(X, y):
    ros = RandomOverSampler()
    X_ros, y_ros = ros.fit_resample(X, y)
    return X_ros, y_ros


def near_miss_1(X, y):
    nm1 = NearMiss(version=1)
    X_nm1, y_nm1 = nm1.fit_resample(X, y)
    return X_nm1, y_nm1

def near_miss_3(X, y):
    nm3 = NearMiss(version=3)
    X_nm3, y_nm3 = nm3.fit_resample(X, y)
    return X_nm3, y_nm3

############################################################
############################################################




############################################################
### III. Rating Features ###################################

### 1. MyAlgo ####################
from surprise import SVD, PredictionImpossible

class MyAlgo(SVD):
    '''
      Best hyperparameters:
        rank = 25, reg = 0.1, biased = True
    '''

    def __init__(self, n_factors=25, n_epochs=30, biased=True, 
                 lr_all=.005, reg_all=0.1, random_state=None, verbose=False):

        SVD.__init__(self, n_factors=n_factors, n_epochs=n_epochs, 
                     biased=biased, lr_all=lr_all, reg_all=reg_all, 
                     random_state=random_state, verbose=verbose)

    def fit(self, trainset):

        SVD.fit(self, trainset)

        return self

    def estimate(self, u, i):

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if known_user and known_item:

            if self.biased:
                est = self.trainset.global_mean
                if known_user:
                    est += self.bu[u]

                if known_item:
                    est += self.bi[i]

                if known_user and known_item:
                    est += np.dot(self.qi[i], self.pu[u])

            else:
                est = np.dot(self.qi[i], self.pu[u])    

        else:
            est = 0
            raise PredictionImpossible('User and item are unknown.')

        return est
    
    def test(self, testset, clip=False, verbose=False):

        predictions = [self.predict(uid,
                                    iid,
                                    r_ui_trans,
                                    clip=clip,
                                    verbose=verbose)
                       for (uid, iid, r_ui_trans) in testset]

        return predictions


### 2. Construct ALS Dataset #####
from surprise import Dataset, Reader, accuracy

def get_als_trainset(train):
    '''
      Input: DataFrame
      Output: Trainset object
    '''
    # Select only the genuine reviews

    train_als = train[(train['label'] == 0)][['user_id', 'prod_id', 'rating']]

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_als[['user_id', 'prod_id', 'rating']], reader)

    return data.build_full_trainset()


def get_als_testset(test):

    testset = list(zip(test['user_id'].values, 
                       test['prod_id'].values, 
                       test['rating'].values))
    return testset


### 3. Rating Features ###########
def get_rating_features(test, algo):

    testset = get_als_testset(test)
    predictions = algo.test(testset)

    actual_rating = []
    pred_rating = []
    is_missing = []
    diff = []
    
    for pred in predictions:
        actual_rating.append(pred.r_ui)
        pred_rating.append(pred.est)
        is_missing.append( int(pred.details['was_impossible']) )
        diff.append(pred.r_ui - pred.est)
    
    rating_features = list(zip(actual_rating, pred_rating, is_missing, diff))

    return rating_features

##################################


############################################################
### IV. User Features ######################################

### 
import numpy as np
import pandas as pd
from collections import Counter

class UserFeature():

    def __init__(self):
        return

    def fit(self, train):
        self.fake_cnt_hist = Counter(train[train['label'] == 1]['user_id'])
        self.avg_fake_cnt = sum(self.fake_cnt_hist.values()) / len(self.fake_cnt_hist)
        return self

    def get_user_features(self, data):
        users = data['user_id'].values
        users_feature = [self.fake_cnt_hist[u] if u in self.fake_cnt_hist else 0 for u in users]
        return np.array(users_feature).reshape([len(users_feature), 1])

    def get_user_features_with_missing_indicator(self, data):
        users = data['user_id'].values

        fake_cnts = np.zeros(len(users))
        is_missing = np.zeros(len(users), dtype='int')

        for i, u in enumerate(users):
            if u in self.fake_cnt_hist:
                fake_cnts[i] = self.fake_cnt_hist[u]
                is_missing[i] = 0
            else:
                fake_cnts[i] = self.avg_fake_cnt
                is_missing[i] = 1

        user_features = list(zip(fake_cnts, is_missing))
        return user_features

##################################



##################################










