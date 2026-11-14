# Project notes

## Objective
Demonstrate an end-to-end NLP workflow for classifying written customer feedback into three sentiment categories.

## Method
- Input: review text
- Feature extraction: TF-IDF word and bigram features
- Classifier: Logistic Regression with balanced class weights
- Evaluation: stratified holdout split, accuracy, classification report, confusion matrix

## Dataset
The checked-in CSV contains synthetic examples for reproducible setup and UI demonstration. It is not collected from customers and must not be represented as real-world evidence.

## Interpretation
A high score on synthetic or templated examples does not establish generalization. Meaningful evaluation requires a representative, independently labelled dataset and error analysis.
