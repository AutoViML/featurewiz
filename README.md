# featurewiz

![banner](featurewiz_logos.png)
<p>

## Update (March 2022)
<ol>
<li><b>featurewiz as of version 0.1.04 or higher can read `feather-format` files at blazing speeds.</b> See example below on how to convert your CSV files to feather. Then you can feed those '.ftr' files to featurewiz and it will read it 10-100X faster!<br>

![feather_example](feather_example.jpg)

<li><b>featurewiz now runs at blazing speeds thanks to using GPU's by default.</b> So if you are running a large data set on Colab and/or Kaggle, make sure you turn on the GPU kernels. featurewiz will automatically detect that GPU is turned on and will utilize XGBoost using GPU-hist. That will ensure it will crunch your datasets even faster. I have tested it with a very large data set and it reduced the running time from 52 mins to 1 minute! That's a 98% reduction in running time using GPU compared to CPU!<br>

## Update (Jan 2022)
<ol>
<li><b>FeatureWiz as of version 0.0.90 or higher is a scikit-learn compatible feature selection transformer.</b> You can perform fit and predict as follows. You will get a Transformer that can select the top variables from your dataset. You can also use it in sklearn pipelines as a Transformer.

```
from featurewiz import FeatureWiz
features = FeatureWiz(corr_limit=0.70, feature_engg='', category_encoders='', 
dask_xgboost_flag=False, nrows=None, verbose=2)
X_train_selected = features.fit_transform(X_train, y_train)
X_test_selected = features.transform(X_test)
features.features  ### provides the list of selected features ###
```

<li><b>Featurewiz is now upgraded with XGBOOST 1.5.1 for DASK for blazing fast performance</b> even for very large data sets! Set `dask_xgboost_flag = True` to run dask + xgboost.
<li><b>Featurewiz now runs with a default setting of `nrows=None`.</b> This means it will run using all rows. But if you want it to run faster, then you can change `nrows` to 1000 or whatever, so it will sample that many rows and run.
<li><b>Featurewiz has lots of new fast model builder functions:</b> that you can use to build highly performant models with the features selected by featurewiz. They are:<br>
1. <b>simple_LightGBM_model()</b> - simple regression and classification with one target label<br>
2. <b>simple_XGBoost_model()</b> - simple regression and classification with one target label<br>
3. <b>complex_LightGBM_model()</b> - more complex multi-label and multi-class models<br>
4. <b>complex_XGBoost_model()</b> - more complex multi-label and multi-class models<br>
5. <b>Stacking_Classifier()</b>: Stacking model that can handle multi-label, multi-class problems<br>
6. <b>Stacking_Regressor()</b>: Stacking model that can handle multi-label, regression problems<br>
7. <b>Blending_Regressor()</b>: Blending model that can handle multi-label, regression problems<br>
</ol>

### One word of CAUTION while installing featurewiz in Kaggle and other environments:
 You must install featurewiz without any dependencies and by ignoring previous installed versions (see below). You MUST execute these TWO steps if you want featurwiz installed and working smoothly.

 ```pip install xlrd```

 ```pip install featurewiz --ignore-installed --no-deps```

## What is featurewiz?
`featurewiz` a new python library for creating and selecting the best features in your data set fast!
`featurewiz` can be used in one or two ways. Both are explained below.

## 1.  Feature Engineering
<p>The first step is not absolutely necessary but it can be used to create new features that may or may not be helpful (be careful with automated feature engineering tools!).<p>
1. <b>Performing Feature Engineering</b>: One of the gaps in open source AutoML tools and especially Auto_ViML has been the lack of feature engineering capabilities that high powered competitions such as Kaggle required. The ability to create "interaction" variables or adding "group-by" features or "target-encoding" categorical variables was difficult and sifting through those hundreds of new features to find best features was difficult and left only to "experts" or "professionals". featurewiz was created to help you in this endeavor.<br>
<p>featurewiz now enables you to add hundreds of such features with a single line of code. Set the "feature_engg" flag to "interactions", "groupby" or "target" and featurewiz will select the best encoders for each of those options and create hundreds (perhaps thousands) of features in one go. Not only that, using the next step, featurewiz will sift through numerous such variables and find only the least correlated and most relevant features to your model. All in one step!.<br>

![feature_engg](feature_engg.jpg)

## 2.  Feature Selection
<p>The second step is Feature Selection. `featurewiz` uses the MRMR (Minimum Redundancy Maximum Relevance) algorithm as the basis for its feature selection. <br>
<b> Why do Feature Selection</b>? Once you have created 100's of new features, you still have three questions left to answer:
1. How do we interpret those newly created features?
2. Which of these features is important and which are useless? How many of them are highly correlated to each other causing redundancy?
3. Does the model overfit now on these new features and perform better or worse than before?
<br>
All are very important questions and featurewiz answers them by using the SULOV method and Recursive XGBoost to reduce features in your dataset to the best "minimum optimal" features for the model.<br>
<p><b>SULOV</b>: SULOV stands for `Searching for Uncorrelated List of Variables`. The SULOV algorithm is based on the Minimum-Redundancy-Maximum-Relevance (MRMR) <a href="https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b">algorithm explained in this article</a> as one of the best feature selection methods. To understand how MRMR works and how it is different from `Boruta` and other feature selection methods, see the chart below. Here "Minimal Optimal" refers to the MRMR and featurewiz kind of algorithms while "all-relevant" refers to Boruta kind of algorithms.<br>

![MRMR_chart](MRMR.png)
<br>
The working of the SULOV algorithm is as follows:
<ol>
<li>Find all the pairs of highly correlated variables exceeding a correlation threshold (say absolute(0.7)).
<li>Then find their MIS score (Mutual Information Score) to the target variable. MIS is a non-parametric scoring method. So its suitable for all kinds of variables and target.
<li>Now take each pair of correlated variables, then knock off the one with the lower MIS score.
<li>What’s left is the ones with the highest Information scores and least correlation with each other.
</ol>

![sulov](SULOV.jpg)

<b>Recursive XGBoost</b>: Once SULOV has selected variables that have high mutual information scores with least less correlation amongst them, we use XGBoost to repeatedly find best features among the remaining variables after SULOV. The Recursive XGBoost method is explained in this chart below.
Here is how it works:
<ol>
<li>Select all variables in data set and the full data split into train and valid sets.
<li>Find top X features (could be 10) on train using valid for early stopping (to prevent over-fitting)
<li>Then take next set of vars and find top X
<li>Do this 5 times. Combine all selected features and de-duplicate them.
</ol>

![xgboost](xgboost.jpg)

<b>Building the simplest and most "interpretable" model</b>: featurewiz represents the "next best" step you must perform after doing feature engineering  since you might have added some highly correlated or even useless features when you use automated feature engineering. featurewiz ensures you have the least number of features needed to build a high performing or equivalent model.

<b>A WORD OF CAUTION:</b> Just because you can engineer new features, doesn't mean you should always create tons of new features. You must make sure you understand what the new features stand for before you attempt to build a model with these (sometimes useless) features. featurewiz displays the SULOV chart which can show you how the 100's of newly created variables added to your dataset are highly correlated to each other and were removed. This will help you understand how feature selection works in featurewiz.

## Table of Contents
<ul>
<li><a href="#background">Background</a></li>
<li><a href="#install">Install</a></li>
<li><a href="#usage">Usage</a></li>
<li><a href="#api">API</a></li>
<li><a href="#maintainers">Maintainers</a></li>
<li><a href="#contributing">Contributing</a></li>
<li><a href="#license">License</a></li>
</ul>

## Background

![background](featurewiz_background.jpg)

To learn more about how featurewiz works under the hood, watch this [video](https://www.youtube.com/embed/ZiNutwPcAU0)<br>

<p>featurewiz was designed for selecting High Performance variables with the fewest steps.

In most cases, featurewiz builds models with 20%-99% fewer features than your original data set with nearly the same or slightly lower performance (this is based on my trials. Your experience may vary).<br>
<p>
featurewiz is every Data Scientist's feature wizard that will:<ol>
<li><b>Automatically pre-process data</b>: you can send in your entire dataframe "as is" and featurewiz will classify and change/label encode categorical variables changes to help XGBoost processing. It classifies variables as numeric or categorical or NLP or date-time variables automatically so it can use them correctly to model.<br>
<li><b>Perform feature engineering automatically</b>: The ability to create "interaction" variables or adding "group-by" features or "target-encoding" categorical variables is difficult and sifting through those hundreds of new features is painstaking and left only to "experts". Now, with featurewiz you can create hundreds or even thousands of new features with the click of a mouse. This is very helpful when you have a small number of features to start with. However, be careful with this option. You can very easily create a monster with this option.
<li><b>Perform feature reduction automatically</b>. When you have small data sets and you know your domain well, it is easy to perhaps do EDA and identify which variables are important. But when you have a very large data set with hundreds if not thousands of variables, selecting the best features from your model can mean the difference between a bloated and highly complex model or a simple model with the fewest and most information-rich features. featurewiz uses XGBoost repeatedly to perform feature selection. You must try it on your large data sets and compare!<br>
<li><b>Explain SULOV method graphically </b> using networkx library so you can see which variables are highly correlated to which ones and which of those have high or low mutual information scores automatically. Just set verbose = 2 to see the graph. <br>
<li><b>Build a fast LightGBM model </b> using the features selected by featurewiz. There is a function called "simple_lightgbm_model" which you can use to build a fast model. It is a new module, so check it out.<br>
</ol>

<b>***  Notes of Gratitude ***</b>:<br>
<ol>
<li><b>Alex Lekov</b> (https://github.com/Alex-Lekov/AutoML_Alex/tree/master/automl_alex) for his DataBunch and encoders modules which are used by the tool (although with some modifications).</li>
<li><b>Category Encoders</b> library in Python : This is an amazing library. Make sure you read all about the encoders that featurewiz uses here: https://contrib.scikit-learn.org/category_encoders/index.html </li>
</ol>

## Install

**Prerequsites:**
<ol>
<li><b>featurewiz is built using xgboost, dask, numpy, pandas and matplotlib</b>. It should run on most Python 3 Anaconda installations. You won't have to import any special libraries other than "dask", "XGBoost" and "networkx" library. Optionally, it uses LightGBM for fast modeling, which it installs automatically. </li>
<li><b>We use "networkx" library for charts and interpretability</b>. <br>But if you don't have these libraries, featurewiz will install those for you automatically.</li>
</ol>
- [Anaconda](https://docs.anaconda.com/anaconda/install/)

To clone featurewiz, it is better to create a new environment, and install the required dependencies:

To install from PyPi:

```
conda create -n <your_env_name> python=3.7 anaconda
conda activate <your_env_name> # ON WINDOWS: `source activate <your_env_name>`
pip install featurewiz --ignore-installed --no-deps
or
pip install git+https://github.com/AutoViML/featurewiz.git
```

To install from source:

```
cd <featurewiz_Destination>
git clone git@github.com:AutoViML/featurewiz.git
# or download and unzip https://github.com/AutoViML/featurewiz/archive/master.zip
conda create -n <your_env_name> python=3.7 anaconda
conda activate <your_env_name> # ON WINDOWS: `source activate <your_env_name>`
cd featurewiz
pip install -r requirements.txt
```

## Usage

As of Jan 2022, you now invoke featurewiz as a scikit-learn compatible fit and predict transformer pipeline. See syntax below.

```
from featurewiz import FeatureWiz
features = FeatureWiz(corr_limit=0.70, feature_engg='', category_encoders='', dask_xgboost_flag=False, nrows=None, verbose=2)
X_train_selected = features.fit_transform(X_train, y_train)
X_test_selected = features.transform(X_test)
features.features  ### provides the list of selected features ###
```

Alternatively, you can continue to use the existing featurewiz function as it is now:
```
import featurewiz as FW
```

Load a data set (any CSV or text file) into a Pandas dataframe and give it the name of the target(s) variable. If you have more than one target, it will handle multi-label targets too. Just give it a list of variables in that case. If you don't have a dataframe, you can simply enter the name and path of the file to load into featurewiz:

```
outputs = FW.featurewiz(dataname, target, corr_limit=0.70, verbose=2, sep=',', 
		header=0, test_data='',feature_engg='', category_encoders='',
		dask_xgboost_flag=False, nrows=None)
```

`outputs`: There will always be multiple objects in output. The objects in that tuple can vary:
1. "features" and "train": It be a list (of selected features) and one dataframe (if you sent in train only)
2. "trainm" and "testm": It can be two dataframes when you send in both test and train but with selected features.
<ol>
<li>Both the selected features and dataframes are ready for you to now to do further modeling.
<li>Featurewiz works on any multi-class, multi-label data Set. So you can have as many target labels as you want.
<li>You don't have to tell Featurewiz whether it is a Regression or Classification problem. It will decide that automatically.
</ol>

## API

**Arguments**

- `dataname`: could be a datapath+filename or a dataframe. It will detect whether your input is a filename or a dataframe and load it automatically.
- `target`: name of the target variable in the data set.
- `corr_limit`: if you want to set your own threshold for removing variables as highly correlated, then give it here. The default is 0.7 which means variables less than -0.7 and greater than 0.7 in pearson's correlation will be candidates for removal.
- `verbose`: This has 3 possible states:
  - `0` limited output. Great for running this silently and getting fast results.
  - `1` more verbiage. Great for knowing how results were and making changes to flags in input.
  - `2` SULOV charts and output. Great for finding out what happens under the hood for SULOV method.
- `test_data`: If you want to transform test data in the same way you are transforming dataname, you can.
    test_data could be the name of a datapath+filename or a dataframe. featurewiz will detect whether
        your input is a filename or a dataframe and load it automatically. Default is empty string.
- `feature_engg`: You can let featurewiz select its best encoders for your data set by setting this flag
    for adding feature engineering. There are three choices. You can choose one, two or all three.
    - `interactions`: This will add interaction features to your data such as x1*x2, x2*x3, x1**2, x2**2, etc.
    - `groupby`: This will generate Group By features to your numeric vars by grouping all categorical vars.
    - `target`:  This will encode and transform all your categorical features using certain target encoders.<br>
    Default is empty string (which means no additional features)
- `category_encoders`: Instead of above method, you can choose your own kind of category encoders from the list below.
    Recommend you do not use more than two of these. Featurewiz will automatically select only two from your list. Default is empty string (which means no encoding of your categorical features)<br> These descriptions are derived from the excellent <a href="https://contrib.scikit-learn.org/category_encoders/"> category_encoders</a> python library. Please check it out!
    - `HashingEncoder`: HashingEncoder is a multivariate hashing implementation with configurable dimensionality/precision. The advantage of this encoder is that it does not maintain a dictionary of observed categories. Consequently, the encoder does not grow in size and accepts new values during data scoring by design.
    - `SumEncoder`: SumEncoder is a Sum contrast coding for the encoding of categorical features.
    - `PolynomialEncoder`: PolynomialEncoder is a Polynomial contrast coding for the encoding of categorical features.
    - `BackwardDifferenceEncoder`: BackwardDifferenceEncoder is a Backward difference contrast coding for encoding categorical variables.
    - `OneHotEncoder`: OneHotEncoder is the traditional Onehot (or dummy) coding for categorical features. It produces one feature per category, each being a binary.
    - `HelmertEncoder`: HelmertEncoder uses the Helmert contrast coding for encoding categorical features.
    - `OrdinalEncoder`: OrdinalEncoder uses Ordinal encoding to designate a single column of integers to represent the categories in your data. Integers however start in the same order in which the categories are found in your dataset. If you want to change the order, just sort the column and send it in for encoding.
    - `FrequencyEncoder`: FrequencyEncoder is a count encoding technique for categorical features. For a given categorical feature, it replaces the names of the categories with the group counts of each category.
    - `BaseNEncoder`: BaseNEncoder encodes the categories into arrays of their base-N representation. A base of 1 is equivalent to one-hot encoding (not really base-1, but useful), a base of 2 is equivalent to binary encoding. N=number of actual categories is equivalent to vanilla ordinal encoding.
    - `TargetEncoder`: TargetEncoder performs Target encoding for categorical features. It supports following kinds of targets: binary and continuous. For multi-class targets it uses a PolynomialWrapper.
    - `CatBoostEncoder`: CatBoostEncoder performs CatBoost coding for categorical features. It supports the following kinds of targets: binary and continuous. For polynomial target support, it uses a PolynomialWrapper. This is very similar to leave-one-out encoding, but calculates the values “on-the-fly”. Consequently, the values naturally vary during the training phase and it is not necessary to add random noise.
    - `WOEEncoder`: WOEEncoder uses the Weight of Evidence technique for categorical features. It supports only one kind of target: binary. For polynomial target support, it uses a PolynomialWrapper. It cannot be used for Regression.
    - `JamesSteinEncoder`: JamesSteinEncoder uses the James-Stein estimator. It supports 2 kinds of targets: binary and continuous. For polynomial target support, it uses PolynomialWrapper.
    For feature value i, James-Stein estimator returns a weighted average of:
    The mean target value for the observed feature value i.
    The mean target value (regardless of the feature value).
    - `dask_xgboost_flag`: Default is False. Set to True to use dask_xgboost estimator. You can turn it off if it gives an error. Then it will use pandas and regular xgboost to do the job.
    - `nrows`: default `None`. You can set the number of rows to read from your datafile if it is too large to fit into either dask or pandas. But you won't have to if you use dask. 
**Return values**
-   `outputs`: Output is always a tuple. We can call our outputs in that tuple: out1 and out2.
    -   `out1` and `out2`: If you sent in just one dataframe or filename as input, you will get:
        - 1. `features`: It will be a list (of selected features) and
        - 2. `trainm`: It will be a dataframe (if you sent in a file or dataname as input)
    -   `out1` and `out2`: If you sent in two files or dataframes (train and test), you will get:
        - 1. `trainm`: a modified train dataframe with engineered and selected features from dataname and
        - 2. `testm`: a modified test dataframe with engineered and selected features from test_data.

## Maintainers

* [@AutoViML](https://github.com/AutoViML)

## Contributing

See [the contributing file](CONTRIBUTING.md)!

PRs accepted.

## License

Apache License 2.0 © 2020 Ram Seshadri

## DISCLAIMER
This project is not an official Google project. It is not supported by Google and Google specifically disclaims all warranties as to its quality, merchantability, or fitness for a particular purpose.
