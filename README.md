# Spam Detection

A simple spam detection web app built with Streamlit, scikit-learn, and TF-IDF text vectorization. The project trains a Naive Bayes classifier on a labeled SMS dataset and exposes a web interface for classifying messages as spam or not spam.

## Project Structure

- `data/spam.csv` - Raw SMS dataset used to train the model.
- `models/spam_model.joblib` - Trained Naive Bayes model.
- `models/tfidf_vectorizer.joblib` - Saved TF-IDF vectorizer for text preprocessing.
- `src/train_model.py` - Script for training the model and generating the saved artifacts.
- `src/spam_app.py` - Streamlit application for user input and spam prediction.
- `requirements.txt` - Python dependencies.

## Features

- Text preprocessing with lowercase conversion and punctuation removal.
- TF-IDF vectorization.
- Multinomial Naive Bayes classification.
- Streamlit interface for entering one or multiple messages.
- Results display with predictions, spam probability, and a chart.

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- streamlit
- joblib
- altair

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the model

If you want to retrain the model from the dataset, run:

```bash
python src/train_model.py
```

This script reads `data/spam.csv`, trains the Naive Bayes classifier, evaluates it on validation and test splits, and saves the model artifacts to the `models/` folder.

### 2. Run the Streamlit app

Start the app with:

```bash
streamlit run src/spam_app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`) and enter one or more messages to classify.

## Notes

- The app accepts multiple messages separated by new lines.
- The prediction results show whether each message is `Spam` or `Not Spam` along with the probability of spam.
- The model and vectorizer are loaded from the `models/` folder when the app starts.

## License

This project is provided as-is for demonstration and learning purposes.
