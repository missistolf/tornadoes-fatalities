# Tornadoes Fatalities Prediction

## Introduction

This is a Machine Learning script that first classifies whether a tornado will or not cause fatalities, and later predicts the specific number of fatalities caused by it. The splitting in two different models was made due to the strong class imbalance present.

## Data

The tornado's data can be found in [here](input/us_tornado_dataset_1950_2021.csv)

## Running the script

One can run this experiment's script through the command line:

```bash
$python3 tornadoes-forecast.py
```

Plots will be shown while running the script, so they need to be closed for the script to keep running. The [output directory](output/) will be rewritten with the plots, and the terminal should give one all relevant metrics and other information printed.

If one desires to run hyperparameter tuning again, remove all files in [model directory](model/). This is where the trained classifier and regressor are stored as pickle files.

## Requirements

In order to install all requirements, please run:

```
pip install -r requirements.txt
```



