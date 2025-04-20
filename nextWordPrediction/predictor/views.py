from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'trained_language_model.h5')
TEXT_PATH = os.path.join(os.path.dirname(__file__), '..', 'friends1.txt')

def load_tokenizer_and_params():
    with open(TEXT_PATH, 'r', encoding='utf-8') as file:
        text = file.read()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1
    input_sequences = []
    for line in text.split('\n'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    max_sequence_len = max([len(seq) for seq in input_sequences])
    return tokenizer, total_words, max_sequence_len

model = load_model(MODEL_PATH)
tokenizer, total_words, max_sequence_len = load_tokenizer_and_params()

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def generate(request):
    if request.method == 'POST':
        if request.content_type == 'application/json':
            import json
            data = json.loads(request.body)
            seed_text = data.get('seed_text', '')
            next_words = int(data.get('next_words', 1))
        else:
            seed_text = request.POST.get('seed_text', '')
            next_words = int(request.POST.get('next_words', 1))
        result = seed_text
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([result])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            predicted = np.argmax(model.predict(token_list), axis=-1)
            output_word = ''
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            if output_word == '':
                break
            result += ' ' + output_word
        return JsonResponse({'generated': result})
    else:
        return HttpResponse('Invalid request', status=400)
