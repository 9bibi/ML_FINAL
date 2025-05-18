# ML Tag Prediction 

A FastAPI web service that predicts tags for programming questions using multi-label classification with Naive Bayes.

## Features
- Multi-label classification for StackOverflow-style questions
- TF-IDF text vectorization
- Naive Bayes prediction model
- Real-time predictions via web interface
- Confidence scores for each predicted tag

## Installation

1. Clone the repository:
```bash
git clone https://github.com/9bibi/ML_FINAL.git
cd ML_FINAL
```
2. Intsall dependencies:
   
pip install -r ../requirements.txt

## Model Training
The model was trained using:

- Scikit-learn's MultiOutputClassifier
- MultinomialNB (Naive Bayes)
- TF-IDF vectorization
- Multi-label binarization

See the collab notebook for details: https://colab.research.google.com/drive/1XTiOgU7WK98_Bbs8KV_O7k0XgvdwPor2?usp=sharing
