from music21 import converter, instrument, note, chord
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
import numpy as np
import pickle
import os


class MusicGenerator(object):
    def __init__(self, data_folder, sequence_length=10, weights_file=None):
        self.data_folder = data_folder
        self.weights_file = weights_file
        self.sequence_length = sequence_length

        if os.path.exists('../songs/notes.pickle'):
            with open('../songs/notes.pickle', 'rb') as f:
                self.notes = pickle.load(f)
        else:
            self.notes = self.retrieve_notes()

        self.n_vocab = len(set(self.notes))
        self.note_to_int, self.int_to_note = self.generate_dictionaries()
        self.X, self.y = self.generate_data()
        self.rnn = self.network_definition()
        self.compile()

    def retrieve_notes(self):
        notes = []
        for file in os.listdir(self.data_folder):
            midi = converter.parse(self.data_folder + file)
            try:
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

        with open('data/notes', 'wb') as filepath:
            pickle.dump(notes, filepath)

        return notes

    def generate_dictionaries(self):
        indexes = sorted(set(item for item in self.notes))
        note_to_int = dict((value, i) for i, value in enumerate(indexes))
        int_to_note = dict((str(i), value) for i, value in enumerate(indexes))
        return note_to_int, int_to_note

    def generate_data(self):
        X = []
        y = []
        for i in range(0, len(self.notes) - self.sequence_length, 1):
            xi = self.notes[i:i + self.sequence_length]
            yi = self.notes[i + self.sequence_length]
            X.append([self.note_to_int[char] for char in xi])
            y.append(self.note_to_int[yi])
        X = np.reshape(X, (len(X), self.sequence_length, 1))
        X = X / float(self.n_vocab)
        y = np_utils.to_categorical(y)
        return X, y

    def network_definition(self):
        model = Sequential()
        model.add(LSTM(
            512,
            input_shape=(self.X.shape[1], self.X.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(self.n_vocab))
        model.add(Activation('softmax'))
        return model

    def fit(self, epochs=200, batch_size=64):
        filepath = "weights-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint]
        self.rnn.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1)

    def predict_random(self, length=500):
        initial_sample = np.random.randint(0, len(self.X) - 1)
        print(initial_sample)
        pattern = self.X[initial_sample]

        ypred = []
        for note_index in range(length):
            Xpred = np.reshape(pattern, (1, len(pattern), 1))

            prediction = self.rnn.predict(Xpred)
            index = np.argmax(prediction)
            ypred.append(self.int_to_note[str(index)])

            pattern = list(pattern)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
            pattern = np.asarray(pattern)

        return ypred

    def save(self):
        model_json = self.rnn.to_json()
        with open('../models/Last_model.json', 'w') as json_file:
            json_file.write(model_json)
        self.rnn.save_weights("../models/last_model_latest_weights_and_arch.hdf5")

    def compile(self):
        if self.weights_file and os.path.exists(self.weights_file):
            print('weights loaded')
            self.rnn.load_weights(self.weights_file)
        self.rnn.summary()
        self.rnn.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
