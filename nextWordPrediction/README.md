# Next Word Prediction Django App

This project is a web based application that uses a trained deep learning model to predict the next word(s) given a seed phrase. The app is built with Django and TensorFlow/Keras.
--------------------------------------------------------------------------------
## Features
- Predict the next word(s) given a seed phrase
- Web UI (see `templates/`)
- Uses a pre-trained Keras model (`trained_language_model.h5`)
-------------------------------------------------------------------------------
## Requirements
- Python 3.8+
- Django
- numpy
- tensorflow

Install dependencies:
```bash
pip install -r requirements.txt
```
-------------------------------------------------------------------------------
## Running the Application
1. Start the Django server:
   ```bash
   python manage.py runserver
   ```
2. Open browser at [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
3. Enter a seed phrase and number of words to predict.
--------------------------------------------------------------------------------
## File Structure
- `predictor/` - Django app with views, urls, and templates
- `templates/` - HTML templates for the web UI
- `trained_language_model.h5` - Pre-trained Keras model
- `friends1.txt` - Text corpus used for training
- `training_model.py` - Script to train or retrain the model
--------------------------------------------------------------------------------