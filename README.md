# Twitter US Airline Sentiment Analysis

A Natural Language Processing (NLP) project to analyze sentiments of tweets about US airlines.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Project Name: Twitter US Airline Sentiment Analysis.
Team:
- Ismail Sherif
- Ahmed Mahgoub
- Mina Hany Ibrahim
- Omar Ehab

Description: A Natural Language Processing (NLP) project to analyze sentiments of tweets about US airlines.

Installation: [Click Here](#installation)

Models in Project: The project uses 4 types of machine learning and one deep learning model (LSTM or RNN) for different tasks.
- Random Forest
- Logistic Regression
- Support Vector Machine
- Naive Bayes
- RNN (LSTM/GRU)

Results The performance should be above 80% and calculate the ROC-Curve, a confusion matrix and (accuracy, precision, Recall and F1-score).

Contact:
- Ismail Sherif: ismailsherifwork@gmail.com
- Ahmed Mahgoub: aed7862@gmail.com
- Omar Ehab: oehab7683@gmail.com
- Mina Hany Ibrahim: mh6639272@gmail.com

## Dataset

- **Source:** [Kaggle - Twitter US Airline Sentiment](https://www.kaggle.com/datasets/mdraselsarker/twitter-us-airline-sentiment-analysis)
- **Features:** Tweet text, airline, sentiment label, etc.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/thisismrismail/twitter-us-airline-sentiment-analysis.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Download and place the dataset in the `data/` directory.
2. Run the main script:
    ```bash
    python main.py
    ```

## Project Structure

```
.
├── data/
├── src/
├── 
├── notebooks/
    └── analysis_visualization/
    └── models/
    └── preprocessing/
├── requirements.txt
└── README.md
```

## Model Details

- Preprocessing: Tokenization, stopword removal, etc.
- Models: Logistic Regression, SVM, or deep learning models.
- Evaluation metrics: Accuracy, F1-score, confusion matrix.

## Results

- Summary of model performance.
- Example predictions.

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.
