from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
# import xgboost as xgb
import lightgbm as lgb


class KNN(KNeighborsClassifier):
    def __init__(self, *params, **kwargs):
        super(KNeighborsClassifier, self).__init__(*params, **kwargs)


class SGDC(SGDClassifier):
    def __init__(self, *params, **kwargs):
        super(SGDClassifier, self).__init__(*params, **kwargs)


class XGBoost:
    def __call__(self):
        # return xgb
        raise NotImplementedError

class LightGBM:
    def __call__(self):
        return lgb


def baseline(train, val, criterion):
    x_train, y_train = train
    x_val, y_val = val

    models = {'knn': KNN(), 'sgdc': SGDC(), 'lgb': LightGBM()}
    results = dict(models)

    for model_name, model in models.items():
        model.fit(x_train, y_train)

        results[model_name] = (criterion(y_train, model.predict(x_train)),
                               criterion(y_val, model.predict(x_val)))

    return results