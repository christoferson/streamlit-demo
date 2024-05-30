import cmn_settings
import boto3
import os
import logging
from contextlib import closing

from botocore.exceptions import BotoCoreError, ClientError

AWS_REGION = cmn_settings.AWS_REGION

polly = boto3.client("polly", region_name=AWS_REGION)
comprehend = boto3.client("comprehend", region_name=AWS_REGION)

language_voice_map = {
    "ja": "Mizuki",
    "en": "Joanna",
    "ko": "Seoyeon",
    "es": "Penelope",
    "nl": "Lotte",
}

def synthesize_speech(text, language):
    try:

        voice_id = language_voice_map.get(language)
        if voice_id == None:
            voice_id = "Joanna"

        response = polly.synthesize_speech(Text=text, OutputFormat="mp3", VoiceId=voice_id)
    except (BotoCoreError, ClientError) as error:
        logging.error(error)
        raise error

    if "AudioStream" not in response:
        raise Exception("Unable to get AudioStream")

    # Note: Closing the stream is important because the service throttles on the
    # number of parallel connections. Here we are using contextlib.closing to
    # ensure the close method of the stream object will be called automatically
    # at the end of the with statement's scope.
    audio_stream = response["AudioStream"]
    audio_binary = None
    with closing(audio_stream) as stream:
        try:
            audio_binary = stream.read()
        except IOError as error:
            print(error)
            raise Exception("Unable to get Stream Binary")

    return audio_binary        

def detect_dominant_language(text):
    try:
        response = comprehend.detect_dominant_language(Text=text)
        languages = response["Languages"]
        #logger.info("Detected %s languages.", len(languages))
    except ClientError:
        logging.exception("Couldn't detect languages.")
        raise
    else:
        return languages