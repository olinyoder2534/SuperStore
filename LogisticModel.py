

import sys
import dataiku
import numpy as np
import pandas as pd
import sklearn as sk
import dataiku.core.pandasutils as pdu
from dataiku.doctor.preprocessing import PCA
from collections import defaultdict, Counter

pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

#preprocessing
preparation_steps = []
preparation_output_schema = {'columns': [{'name': 'Id', 'type': 'bigint'}, {'name': 'Year_Birth', 'type': 'bigint'}, {'name': 'Education', 'type': 'string'}, {'name': 'Marital_Status', 'type': 'string'}, {'name': 'Income', 'type': 'bigint'}, {'name': 'Kidhome', 'type': 'bigint'}, {'name': 'Teenhome', 'type': 'bigint'}, {'name': 'Dt_Customer', 'type': 'string'}, {'name': 'Recency', 'type': 'bigint'}, {'name': 'MntWines', 'type': 'bigint'}, {'name': 'MntFruits', 'type': 'bigint'}, {'name': 'MntMeatProducts', 'type': 'bigint'}, {'name': 'MntFishProducts', 'type': 'bigint'}, {'name': 'MntSweetProducts', 'type': 'bigint'}, {'name': 'MntGoldProds', 'type': 'bigint'}, {'name': 'NumDealsPurchases', 'type': 'bigint'}, {'name': 'NumWebPurchases', 'type': 'bigint'}, {'name': 'NumCatalogPurchases', 'type': 'bigint'}, {'name': 'NumStorePurchases', 'type': 'bigint'}, {'name': 'NumWebVisitsMonth', 'type': 'bigint'}, {'name': 'Response', 'type': 'bigint'}, {'name': 'Complain', 'type': 'bigint'}, {'name': 'daysCust', 'type': 'bigint'}], 'userModified': False}

ml_dataset_handle = dataiku.Dataset('superStoreCleaned')
ml_dataset_handle.set_preparation_steps(preparation_steps, preparation_output_schema)
%time ml_dataset = ml_dataset_handle.get_dataframe(limit = 100000)

print ('Base data has %i rows and %i columns' % (ml_dataset.shape[0], ml_dataset.shape[1]))
# Five first records",
ml_dataset.head(5)


ml_dataset = ml_dataset[['MntWines', 'Marital_Status', 'NumWebPurchases', 'Income', 'Teenhome', 'MntFruits', 'daysCust', 'Year_Birth', 'NumCatalogPurchases', 'Response', 'MntMeatProducts', 'NumWebVisitsMonth', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'Recency', 'Education', 'NumStorePurchases', 'Complain', 'Kidhome', 'NumDealsPurchases']]

# astype('unicode') does not work as expected

def coerce_to_unicode(x):
    if sys.version_info < (3, 0):
        if isinstance(x, str):
            return unicode(x,'utf-8')
        else:
            return unicode(x)
    else:
        return str(x)

#convert categorical variables into unicode & numeric into floats
categorical_features = ['Marital_Status', 'Education']
numerical_features = ['MntWines', 'NumWebPurchases', 'Income', 'Teenhome', 'MntFruits', 'daysCust', 'Year_Birth', 'NumCatalogPurchases', 'MntMeatProducts', 'NumWebVisitsMonth', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'Recency', 'NumStorePurchases', 'Complain', 'Kidhome', 'NumDealsPurchases']
text_features = []
from dataiku.doctor.utils import datetime_to_epoch
for feature in categorical_features:
    ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
for feature in text_features:
    ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
for feature in numerical_features:
    if ml_dataset[feature].dtype == np.dtype('M8[ns]') or (hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')):
        ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
    else:
        ml_dataset[feature] = ml_dataset[feature].astype('double')

#remap Reponse (dependent) variable
target_map = {'0': 0, '1': 1}
ml_dataset['__target__'] = ml_dataset['Response'].map(str).map(target_map)
del ml_dataset['Response']


# Remove rows for which the target is unknown.
ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]

ml_dataset['__target__'] = ml_dataset['__target__'].astype(np.int64)

#train-test split
train, test = pdu.split_train_valid(ml_dataset, prop=0.8)
print ('Train data has %i rows and %i columns' % (train.shape[0], train.shape[1]))
print ('Test data has %i rows and %i columns' % (test.shape[0], test.shape[1]))


drop_rows_when_missing = []
impute_when_missing = [{'feature': 'MntWines', 'impute_with': 'MEAN'}, {'feature': 'NumWebPurchases', 'impute_with': 'MEAN'}, {'feature': 'Income', 'impute_with': 'MEAN'}, {'feature': 'Teenhome', 'impute_with': 'MEAN'}, {'feature': 'MntFruits', 'impute_with': 'MEAN'}, {'feature': 'daysCust', 'impute_with': 'MEAN'}, {'feature': 'Year_Birth', 'impute_with': 'MEAN'}, {'feature': 'NumCatalogPurchases', 'impute_with': 'MEAN'}, {'feature': 'MntMeatProducts', 'impute_with': 'MEAN'}, {'feature': 'NumWebVisitsMonth', 'impute_with': 'MEAN'}, {'feature': 'MntFishProducts', 'impute_with': 'MEAN'}, {'feature': 'MntSweetProducts', 'impute_with': 'MEAN'}, {'feature': 'MntGoldProds', 'impute_with': 'MEAN'}, {'feature': 'Recency', 'impute_with': 'MEAN'}, {'feature': 'NumStorePurchases', 'impute_with': 'MEAN'}, {'feature': 'Complain', 'impute_with': 'MEAN'}, {'feature': 'Kidhome', 'impute_with': 'MEAN'}, {'feature': 'NumDealsPurchases', 'impute_with': 'MEAN'}]

# Features for which we drop rows with missing values"
for feature in drop_rows_when_missing:
    train = train[train[feature].notnull()]
    test = test[test[feature].notnull()]
    print ('Dropped missing records in %s' % feature)

# Features for which we impute missing values"
for feature in impute_when_missing:
    if feature['impute_with'] == 'MEAN':
        v = train[feature['feature']].mean()
    elif feature['impute_with'] == 'MEDIAN':
        v = train[feature['feature']].median()
    elif feature['impute_with'] == 'CREATE_CATEGORY':
        v = 'NULL_CATEGORY'
    elif feature['impute_with'] == 'MODE':
        v = train[feature['feature']].value_counts().index[0]
    elif feature['impute_with'] == 'CONSTANT':
        v = feature['value']
    train[feature['feature']] = train[feature['feature']].fillna(v)
    test[feature['feature']] = test[feature['feature']].fillna(v)
    print ('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))

LIMIT_DUMMIES = 100

categorical_to_dummy_encode = ['Marital_Status', 'Education']

# Only keep the top 100 values
def select_dummy_values(train, features):
    dummy_values = {}
    for feature in categorical_to_dummy_encode:
        values = [
            value
            for (value, _) in Counter(train[feature]).most_common(LIMIT_DUMMIES)
        ]
        dummy_values[feature] = values
    return dummy_values

DUMMY_VALUES = select_dummy_values(train, categorical_to_dummy_encode)

def dummy_encode_dataframe(df):
    for (feature, dummy_values) in DUMMY_VALUES.items():
        for dummy_value in dummy_values:
            dummy_name = u'%s_value_%s' % (feature, coerce_to_unicode(dummy_value))
            df[dummy_name] = (df[feature] == dummy_value).astype(float)
        del df[feature]
        print ('Dummy-encoded feature %s' % feature)

dummy_encode_dataframe(train)

dummy_encode_dataframe(test)


#rescale
rescale_features = {'MntWines': 'AVGSTD', 'NumWebPurchases': 'AVGSTD', 'Income': 'AVGSTD', 'Teenhome': 'AVGSTD', 'MntFruits': 'AVGSTD', 'daysCust': 'AVGSTD', 'Year_Birth': 'AVGSTD', 'NumCatalogPurchases': 'AVGSTD', 'MntMeatProducts': 'AVGSTD', 'NumWebVisitsMonth': 'AVGSTD', 'MntFishProducts': 'AVGSTD', 'MntSweetProducts': 'AVGSTD', 'MntGoldProds': 'AVGSTD', 'Recency': 'AVGSTD', 'NumStorePurchases': 'AVGSTD', 'Complain': 'AVGSTD', 'Kidhome': 'AVGSTD', 'NumDealsPurchases': 'AVGSTD'}
for (feature_name, rescale_method) in rescale_features.items():
    if rescale_method == 'MINMAX':
        _min = train[feature_name].min()
        _max = train[feature_name].max()
        scale = _max - _min
        shift = _min
    else:
        shift = train[feature_name].mean()
        scale = train[feature_name].std()
    if scale == 0.:
        del train[feature_name]
        del test[feature_name]
        print ('Feature %s was dropped because it has no variance' % feature_name)
    else:
        print ('Rescaled %s' % feature_name)
        train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
        test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale

#modeling
X_train = train.drop('__target__', axis=1)
X_test = test.drop('__target__', axis=1)

y_train = np.array(train['__target__'])
y_test = np.array(test['__target__'])

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty="l2",random_state=1337)

clf.class_weight = "balanced"

%time clf.fit(X_train, y_train)

#testing
%time _predictions = clf.predict(X_test)
%time _probas = clf.predict_proba(X_test)
predictions = pd.Series(data=_predictions, index=X_test.index, name='predicted_value')
cols = [
    u'probability_of_value_%s' % label
    for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
]
probabilities = pd.DataFrame(data=_probas, index=X_test.index, columns=cols)

# Build scored dataset
results_test = X_test.join(predictions, how='left')
results_test = results_test.join(probabilities, how='left')
results_test = results_test.join(test['__target__'], how='left')
results_test = results_test.rename(columns= {'__target__': 'Response'})


#results
from dataiku.doctor.utils.metrics import mroc_auc_score
y_test_ser = pd.Series(y_test)
 
print ('AUC value:', mroc_auc_score(y_test_ser, _probas))

inv_map = { target_map[label] : label for label in target_map}
predictions.map(inv_map)

