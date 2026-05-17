# Part 3: NLP and Sequence Modeling Mini Project

This repository contains an end-to-end Natural Language Processing (NLP) pipeline designed to classify customer support text data. 

## Project Overview
The project compares two distinct approaches to NLP text classification:
1. **Traditional Machine Learning:** Utilizing TF-IDF Vectorization combined with a Logistic Regression baseline model.
2. **Deep Sequence Modeling:** Utilizing Tokenizer sequencing, word embeddings, and a Long Short-Term Memory (LSTM) neural network to capture sequential context.

## Repository Structure
- `notebook.ipynb`: Contains the full implementation steps for data preprocessing, vectorization, and model training.
- `requirements.txt`: Python package dependencies.
- `results/model_evaluation.png`: Contains the training and validation accuracy/loss plots for the LSTM model.
- `results/sample_predictions.txt`: A text file output showing side-by-side actual vs. predicted classifications on unseen data.

## Theoretical Insights
This project explores why text must be vectorized (translating human language into mathematical representations) and demonstrates how LSTMs overcome the vanishing gradient problems associated with traditional RNNs by utilizing gating mechanisms to retain long-term memory dependencies.
