### this defines some of the global settings for encoder names in one place ####
from category_encoders import HashingEncoder, SumEncoder, PolynomialEncoder, BackwardDifferenceEncoder
from category_encoders import OneHotEncoder, HelmertEncoder, OrdinalEncoder, CountEncoder, BaseNEncoder
from category_encoders import TargetEncoder, CatBoostEncoder, WOEEncoder, JamesSteinEncoder
from category_encoders.glmm import GLMMEncoder
from sklearn.preprocessing import LabelEncoder
from category_encoders.wrapper import PolynomialWrapper
from .encoders import FrequencyEncoder
#################################################################################
def init():
    global cat_encoders_names
    cat_encoders_names = {
        'HashingEncoder': [HashingEncoder,'https://contrib.scikit-learn.org/category_encoders/hashing.html'],
        'SumEncoder': [SumEncoder,'https://contrib.scikit-learn.org/category_encoders/sum.html'],
        'PolynomialEncoder': [PolynomialEncoder,'https://contrib.scikit-learn.org/category_encoders/polynomial.html'],
        'BackwardDifferenceEncoder': [BackwardDifferenceEncoder,'https://contrib.scikit-learn.org/category_encoders/backward_difference.html'],
        'OneHotEncoder': [OneHotEncoder,'https://contrib.scikit-learn.org/category_encoders/onehot.html'],
        'HelmertEncoder': [HelmertEncoder,'https://contrib.scikit-learn.org/category_encoders/helmert.html'],
        'OrdinalEncoder': [OrdinalEncoder,'https://contrib.scikit-learn.org/category_encoders/ordinal.html'],
        'BaseNEncoder': [BaseNEncoder,'https://contrib.scikit-learn.org/category_encoders/basen.html'],
        'FrequencyEncoder': [FrequencyEncoder,'https://github.com/Alex-Lekov/AutoML_Alex/blob/master/automl_alex/encoders.py'],
        }

    global target_encoders_names
    target_encoders_names = {
        'TargetEncoder': [TargetEncoder,'https://contrib.scikit-learn.org/category_encoders/targetencoder.html'],
        'CatBoostEncoder': [CatBoostEncoder,'https://contrib.scikit-learn.org/category_encoders/catboost.html'],
        'WOEEncoder': [WOEEncoder,'https://contrib.scikit-learn.org/category_encoders/woe.html'],
        'JamesSteinEncoder': [JamesSteinEncoder,'https://contrib.scikit-learn.org/category_encoders/jamesstein.html'],
        'GLMMEncoder': [GLMMEncoder,'https://contrib.scikit-learn.org/category_encoders/glmm.html'],
        }

    global modeltpe
    modeltpe = ''
    global multi_label
    multi_label = False
#################################################################################
