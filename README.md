# Google Analytics Customer Revenue Prediction

Top-2% solution to the [Google Analytics Customer Revenue Prediction](https://www.kaggle.com/c/ga-customer-revenue-prediction) Kaggle competition on customer spending forecasting.

![features](https://i.postimg.cc/6QKFDSwb/var-importance.png)


## Summary

Predicting future revenues generated by a certain customer is an important e-commerce task. Revenue forecasts are frequently used by marketing departments to refine the targeting and promotional strategies. Machine learning can help to produce accurate predictions for existing customers.  

This project works with a two-year data set with user transactions in the Google Merchandise Store. We perform a thorough data preprocessing and aggregation, engineering features such as recency, frequency and page visit statistics. We build an ensemble of LightGBM models to predict future revenues generated by the existing Google customers. Our solution reaches the RMSE 0.886 on the private leaderboard, placing in the top-2% of the corresponding Kaggle competition.


## Project structure

The project has the following structure:
- `codes/`: Python codes implementing data preprocessing and feature engineering
- `notebooks/`:  Jupyter notebooks covering data preparation, modeling and ensembling
- `data/`: input data (not included due to size constraints and can be downloaded [here](https://www.kaggle.com/c/ga-customer-revenue-prediction/data))
- `oof_preds/`: out-of-fold predictions produced by the trained models within cross-validation
- `submissions/`: test set predictions produced by the trained models


## Working with the repo

To run the project codes, you can create a new virtual environment in `conda`:

```
conda create -n py3 python=3.7
conda activate py3
```

and then install the requirements listed in `requirements.txt`:

```
conda install -n py3 --yes --file requirements.txt
pip install lightgbm
pip install imblearn
```
