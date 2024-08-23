import os
import numpy as np
import pretty_midi


def midi2pianoroll(midi_path, window_size=304, stride=1, fs=32):

    midi_data = pretty_midi.PrettyMIDI(midi_path)


    piano_roll = midi_data.get_piano_roll(fs, pedal_threshold=None)


    piano_roll[piano_roll > 0] = 1


    piano_roll = piano_roll[21:109]

    if piano_roll.shape[1] < window_size:
        pad = window_size - piano_roll.shape[1]
        piano_roll = np.pad(piano_roll, ((0, 0), (0, pad)), mode='constant', constant_values=0)


    segments = []
    for i in range(0, piano_roll.shape[1]-window_size+1, stride):
        segment = piano_roll[:, i: i+window_size]
        segments.append(segment)


    stacked_segments = np.stack(segments, axis=0)

    return stacked_segments


def pianoroll2midi(piano_roll, fs=32, program=0):

    midi = pretty_midi.PrettyMIDI()


    instrument = pretty_midi.Instrument(program=program)


    notes, frames = piano_roll.shape
    for note in range(notes):
        onsets = np.where(np.diff(piano_roll[note, :]) == 1)[0] + 1
        offsets = np.where(np.diff(piano_roll[note, :]) == -1)[0] + 1

        for onset, offset in zip(onsets, offsets):

            start_time = onset / fs
            end_time = offset / fs

            instrument.notes.append(pretty_midi.Note(velocity=60,
                                                     pitch=note + 21,
                                                     start=start_time,
                                                     end=end_time))


    midi.instruments.append(instrument)

    return midi


if __name__ == "__main__":
    from glob import glob


    midi_path = glob('../../datasets/maestro-v3.0.0/2017/*essed_046_PIANO046_MID--AUDIO-split_07-06-17_Piano-e_2-02_wav--3.midi')
    x = midi2pianoroll(midi_path[0], window_size=4096, stride=1000)

    inputs = x.reshape(-1, 88, 64, 64).astype(np.float32)

    midi = pianoroll2midi(x[0])
    midi.write('.midi')