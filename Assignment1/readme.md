# SMS Spam Classification

This project implements a complete machine learning pipeline for classifying SMS messages as **spam** or **ham** using classical text-based models. The workflow covers data preparation, preprocessing, model training, validation, hyperparameter tuning, and final evaluation.

---

## Dataset

The SMS Spam Collection dataset is used, obtained from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

---

## Folder Structure
```
sms_spam_classification/
├── data/
│   └── SMSSpamCollection
├── prepare.ipynb # Preprocessing and data splits
├── train.ipynb # Training, tuning, evaluation
├── train.csv
├── validation.csv
├── test.csv
├── requirements.txt
└── readme.md
```

---

## Method

- Text cleaning: lowercasing, punctuation removal, empty message removal  
- TF-IDF feature extraction  
- Stratified train/validation/test split (70/15/15)  
- Models: Naive Bayes, Logistic Regression, Linear SVM  
- Class imbalance handled using class-weighted loss  
- Model selection based on validation F1 score  

---

## Result

 Evaluation focuses on precision, recall, and F1 score due to class imbalance. Naive Bayes achieved best performance on test set with F1 score 0.938.




