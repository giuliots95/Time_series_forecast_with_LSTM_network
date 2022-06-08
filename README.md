# Time_series_forecast_with_LSTM_network
This repository contains the implementation of a recurrent neural network (LSTM from keras library) with the purpose of forecasting a univariate time series, given its historical records and some covariates. 
The project uses a toy data set, with limited number of predictors (input features) and only one target. The development focuses on:
- data transformation tasks (pandas dataframes to 3D numpy arrays required by recurrent networks) 
- model building using keras HyperModel class
- hyperparameters optimizations trying both HyperBand and an embedded Bayesian optimizer.

## Problem statement
The goal of this project is to build a recurrent neural network that can forecast the partculate matter (pollution indicator), given its past measurements and some varaibles representing weather condition in the same time (air temperature, wind speed...).
Formally, this can be stated as a univariate timeseries forecasting problem with known past and future covariates.

## Mathematical models
This study aims to implement and fine-tune a multi-layer long short time memory network (LSTM). These networks belong to the bigger family of recurrent networks (RNN) and are currently popular choices for timeseries forecasting problems, especially when dealing with multi-input/multi-output problems.
The implementation relies on a sequential model (as suggested by keras) containing a tunable number of LSTM layers. Only the last output layer has been set to a dense layer, with pre-defined output shape and linear activation function (so working as a common 'output' layer in purely dense networks).

## Data transformation 
Transforming a set of time series to match the input requirements of recurrent networks is a no trivial task at all. A part fron the usual train/test split and rescaling tasks, that can more easily be done with some preprocessing utilities from sklearn, one has to first understand and implement the following steps.

### Lagged dataframe
The ensamble of time series needs to be transormed into a unique dataframe of lagged columns. For each value of the target variable to be forecasted, other columns will contain the lagged values of both target variable and covariates.

### Reshaping operation
Multivariate time series are usually saved in indexed tabular data structures, e.g. pandas' dataframes, so they are bidimensional data where each row represents an observation (a sample) and each column contains the numeric value corresponding to a specific feature. On the other hand, recurrent networks process data organized in a 3D fashion. Tensors dimensions are: batch size, timestamps per batch, number of features.
Practically, we have to split the whole amount of samples (rows of the dataframe) into a discrete number of batches of samples, each one containing a fixed number of samples.

## Hypermodels
The most interesting part of this study is the subclassing of keras'HyperModel class, used to customize both the model buidling and the fitting phases.
The hypermodel can contain a variable number of LSTM layers and each layer has its tunable hyperparameters, which are different from the hyperparameters of the other layers. In many implementations available in literature, people are used to define the hyperparameters search space 'a priori', and during the research, to apply the same set of hyperparameters to all layers. For example, once the number of neurons is chosen by the optimizer, the same value is applied to all layers. I wanted to do something more refined, thus expanding the research space, by allowing to each layer (which number can vary depending from another hyperparameter) to take its specific number of neurons. This strategy can be generalized to all hyperparameters involved in the construction of a LSTM layer.

Moreover, each hypermodel is fitted using 'Adam' algorithm with a tunable batch size.

