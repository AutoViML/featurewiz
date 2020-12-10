# featurewiz

![banner](featurewiz_logo.jpg)

Featurewiz is a new python library for selecting the best features in your data set fast!
(featurewiz logo created using Wix)
<p>Two methods are used in this version of featurewiz:<br>

1. SULOV -> SULOV means Searching for Uncorrelated List of Variables. The SULOV method is explained in this chart below. SULOV stands for:  “Searching for Uncorrelated List Of Variables”

Here is a simple way of explaining how it works:
<ol>
<li>Find all the pairs of highly correlated variables exceeding a correlation threshold (say absolute(0.7)).
<li>Then find their MIS score (Mutual Information Score) to the target variable. MIS is a non-parametric scoring method. So its suitable for all kinds of variables and target.
<li>Now take each pair of correlated variables, then knock off the one with the lower MIS score.
<li>What’s left is the ones with the highest Information scores and least correlation with each other.
</ol>


![sulov](SULOV.jpg)

2. Recursive XGBoost: Once SULOV has selected variables that have high mutual information scores with least less correlation amongst them, we use XGBoost to repeatedly find best features among the remaining variables after SULOV. The Recursive XGBoost method is explained in this chart below.
Once have done SULOV method, now select the best variables using XGBoost feature important but apply it recursively to smaller and smaller sets of data in your data set. This is how it works:
<ol>
<li>Select all variables in data set and the full data split into train and valid sets.
<li>Find top X features (could be 10) on train using valid for early stopping (to prevent over-fitting)
<li>Then take next set of vars and find top X
<li>Do this 5 times. Combine all selected features and de-duplicate them.
</ol>


![xgboost](xgboost.jpg)

3. Classification of variables by type: It automatically detects the different types of variables in your data set and converts them to numeric except date-time, NLP and large-text variables. These variables must be properly encoded and transformed (or embedded) into numeric form by you if you want them included in featurewiz selection.<br>

4. Best step after feature engineering: Featurewiz represents the next best step you can perform after doing some feature engineering on your own since you might have added some highly correlated or even wasteful features when you use some automated tools such as featuretools to perform feature engineering. With featurewiz as the last step before you do modeling, you can perform feature selection with featurewiz and the best and least number of features before doing more expensive training and inference.

<p>To upgrade to the best, most stable and full-featured version always do the following: <br>
<code>Use $ pip install featurewiz --upgrade --ignore-installed</code><br>
or
<code>pip install git+https://github.com/AutoViML/featurewiz.git </code><br>

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

To learn more about how featurewiz works under the hood, watch this [video](https://www.youtube.com/embed/ZiNutwPcAU0)<br>

<p>featurewiz was designed for selecting High Performance variables with the fewest steps.

In most cases, featurewiz builds models with 20%-99% fewer features than your original data set with nearly the same or slightly lower performance (this is based on my trials. Your experience may vary).<br>
<p>
featurewiz is every Data Scientist's feature wizard that will:<ol>
<li><b>Automatically pre-process data</b>: you can send in your entire dataframe as is and featurewiz will classify and change/label encode categorical variables changes to help XGBoost processing. That way, you don't have to preprocess your data before using featurewiz<br>
<li><b>Assist you with variable classification</b>: featurewiz classifies variables automatically. This is very helpful when you have hundreds if not thousands of variables since it can readily identify which of those are numeric vs categorical vs NLP text vs date-time variables and so on.<br>
<li><b>Perform feature reduction automatically</b>. When you have small data sets and you know your domain well, it is easy to perhaps do EDA and identify which variables are important. But when you have a very large data set with hundreds if not thousands of variables, selecting the best features from your model can mean the difference between a bloated and highly complex model or a simple model with the fewest and most information-rich features. featurewiz uses XGBoost repeatedly to perform feature selection. You must try it on your large data sets and compare!<br>
<li><b>Explain SULOV method graphically </b> using networkx library so you can see which variables are highly correlated to which ones and which of those have high or low mutual information scores automatically. Just set verbose = 2 to see the graph. <br>
</ol>
featurewiz is built using xgboost, numpy, pandas and matplotlib. It should run on most Python 3 Anaconda installations. You won't have to import any special
libraries other than "XGBoost" and "networkx" library. We use "networkx" library for interpretability. <br>But if you don't have these libraries, featurewiz will install those for you automatically.

## Install

**Prerequsites:**

- [Anaconda](https://docs.anaconda.com/anaconda/install/)

To clone featurewiz, it is better to create a new environment, and install the required dependencies:

To install from PyPi:

```
conda create -n <your_env_name> python=3.7 anaconda
conda activate <your_env_name> # ON WINDOWS: `source activate <your_env_name>`
pip install featurewiz
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

In the same directory, open a Jupyter Notebook and use this line to import the .py file:

```
from featurewiz import featurewiz
```

Load a data set (any CSV or text file) into a Pandas dataframe and give it the name of the target(s) variable. If you have more than one target, it will handle multi-label targets too. Just give it a list of variables in that case. If you don't have a dataframe, you can simply enter the name and path of the file to load into featurewiz:

```
features = featurewiz(
    dataname,
    target,
    corr_limit=0.7,
    verbose=2,
    sep=",",
    header=0)
```

Finally, it returns the list of variables selected.

This list of selected features is ready for you to now to do further modeling.

featurewiz works on any Multi-Class, Multi-Label Data Set. So you can have as many target labels as you want.
You don't have to tell featurwiz whether it is a Regression or Classification problem. It will decide that automatically.

## API

**Arguments**

- `dataname`: could be a datapath+filename or a dataframe. It will detect whether your input is a filename or a dataframe and load it automatically.
- `target`: name of the target variable in the data set.
- `corr_limit`: if you want to set your own threshold for removing variables as highly correlated, then give it here. The default is 0.7 which means variables less than -0.7 and greater than 0.7 in pearson's correlation will be candidates for removal.
- `verbose`: This has 3 possible states:
  - `0` limited output. Great for running this silently and getting fast results.
  - `1` more verbiage. Great for knowing how results were and making changes to flags in input.
  - `2` SULOV charts and output. Great for finding out what happens under the hood for SULOV method.

**Return values**

- `features`: the fewest number of features in your model to make it perform well

## Maintainers

* [@AutoViML](https://github.com/AutoViML)

## Contributing

See [the contributing file](CONTRIBUTING.md)!

PRs accepted.

## License

Apache License 2.0 © 2020 Ram Seshadri

## DISCLAIMER
This project is not an official Google project. It is not supported by Google and Google specifically disclaims all warranties as to its quality, merchantability, or fitness for a particular purpose.
