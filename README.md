# Kaggle Google Analytics Customer Revenue Prediction

Files and codes with our solution to the [Kaggle Google Analytics Customer Revenue Prediction competition](https://www.kaggle.com/c/ga-customer-revenue-prediction).


## Project summary

Predicting future revenue generated by a customer is an importand marketing task. This project uses historical transactional data to predict revenues of Google Merchandise Store customers. 


## Project structure

The project has the follwoing structure:
- `codes/`: jupyter notebooks with codes for different project stages: data preparation, modeling and ensembling.
- `data/`: input data. The folder is not uploaded to Github due to size constraints. The raw data can be downloaded [here](https://www.kaggle.com/c/home-credit-default-risk).
- `oof_preds/`: out-of-fold predictions produced by the train models within cross-validation.
- `submissions/`: test sample predictions produced by the trained models.


## Requirments

To run the project codes, you can create a new virtual environment in `conda`:

```
conda create -n py3 python=3.7
conda activate py3
```

and then install the requirements:

```
conda install -n py3 --yes --file requirements.txt
pip install lightgbm
pip install imblearn
```