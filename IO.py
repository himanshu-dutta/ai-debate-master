#!/usr/bin/env python
# coding: utf-8

import speech_recognition as sr
from gtts import gTTS 

r = sr.Recognizer()
r.pause_threshold = 30

def speech_to_text(filename):
    '''
        Input: Takes a .wav file as input.
        Output: Returns the text recognized from that file.
    '''
    with sr.AudioFile(filename) as source:
        audio_data = r.record(source)
    text = r.recognize_google(audio_data)
    return(text)


def get_speech(ambient_duration):
    '''
        Input: Duration of the ambient noise to be adjusted.
        Output: Returns the audio file recognized.
    '''
    sample_rate = 48000
    chunk_size = 2048
    with sr.Microphone(sample_rate = sample_rate, chunk_size = chunk_size) as source:
        print("Reading...")
        r.adjust_for_ambient_noise(source,duration=ambient_duration)
        audio = r.listen(source)
    return audio


def save_audio(path,audio):
    if str(type(audio)) in "<class 'gtts.tts.gTTS'>":
        audio.save(path+'.wav')
    else:
        with open(path+".wav", "wb") as f:
            f.write(audio.get_wav_data())


def text_to_speech(text):
    '''
        Input: Takes a text/str as input.
        Output: Returns the recognized audio from the text.
    '''
    speech = gTTS(text = text, lang = 'en', slow = False)
    return speech


def recognize_speech(audio):
    '''
            Input: Takes the audio file recognized.
            Output: Returns the text that is recognized.
    '''
    try:
    #r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

