from src.nn_models import MusicGenerator
from music21 import instrument, note, stream, chord


def generate_midi(song, filename):
    offset = 0
    output_notes = []
    for pattern in song:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=filename)


if __name__ == "__main__":
    data_folder = '../songs/Undertale/'
    model = MusicGenerator(data_folder, sequence_length=100, weights_file="../models/weights-44-0.2434-bigger.hdf5")
    print(model.n_vocab)
    #model.fit(epochs=200)
    #model.save()
    #print('predicting')
    #prediction = model.predict_random(50)
    #print('generating')
    #generate_midi(prediction, 'music_generated.mid')
