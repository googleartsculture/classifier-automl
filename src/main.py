#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Main module used to execute HTTP requests
"""

# pylint: disable=unused-argument
# pylint: disable=consider-using-f-string
# pylint: disable=raise-missing-from

# -*- coding: utf-8 -*-
import base64
import os
import re
from functools import wraps
from io import BytesIO
from time import time

from flask import Flask, json, jsonify, request
from flask.logging import logging
from flask_cors import CORS
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from werkzeug.exceptions import (BadRequest, HTTPException, InternalServerError)

from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.cloud import storage

TEST_BUCKET = os.environ.get('TEST_BUCKET', '')
AI_REGION = os.environ.get('AI_REGION', 'europe-west4')
AI_PROJECT = os.environ.get('AUTO_ML_PROJECT', 'cilex-fabricius-workbench-prod')
AI_ENDPOINT_ID = os.environ.get('AI_ENDPOINT_ID', '995559400439545856')

weights = {}


# Define a new get_headers() function for the HTTPException class,
# to return application/json MIME type rather than plain HTML
# @TODO Review how this is being done and implement in a more pythonic way
def get_headers(self, environ=None, scope=None):
  return [('Content-Type', 'application/json')]


HTTPException.get_headers = get_headers


# Define a new get_body() function for the HTTPException class,
# to return json rather than plain HTML
# @TODO Review how this is being done and implement in a more pythonic way
def get_body(self, environ=None, scope=None):
  return json.dumps({
      'success': False,
      'message': self.description,
      'code': self.code
  })


HTTPException.get_body = get_body


# Create a wrapper for ensuring API requests have an application/json MIME type,
# raising a custom BadRequest if they don't
# @TODO Review whether there is a better way to implement
# this.
# A decorator seems like the most sensible, but mayber there is a better way?
def require_json(params=None):
  '''Decorator function to wrap app route functions when we explicity
    want the Content-Type to be application/json. Checks the request.is_json
    and raises a BadRequest exception if not. Also allows for a list of
    required parameters and checks for their existence, rasing a BadRequest if
    any of them are missing.'''
  if params is None:
    params = []

  def require_json_inner(func):

    @wraps(func)
    def func_wrapper(*args, **kwargs):
      if request.is_json:

        json_payload = request.get_json(cache=False)
        for param in params:
          if not json_payload.get(param):
            raise BadRequest(f'The request is missing the {param} parameter')

        return func(json_payload, *args, **kwargs)

      raise BadRequest('The request content type must be application/json')

    return func_wrapper

  return require_json_inner


def create_filename(prediction):
  """Create a filename to save to the bucket"""
  epoc = int(time())
  return '{prediction}/{epoc}-{prediction}.png'.format(prediction=prediction,
                                                       epoc=epoc)


def upload_image(filename, image):
  if TEST_BUCKET is None:
    # Environment not set for storing test images
    return
  storage_client = storage.Client()
  bucket = storage_client.bucket(TEST_BUCKET)
  blob = bucket.blob(filename)
  blob.upload_from_string(image, content_type='image/png')
  return


# Instanciate the flask app and enable CORS on it
app = Flask(__name__)
CORS(app)


@app.route('/_ah/warmup')
def warmup():
  return '', 200, {}


@app.route('/classification', methods=['POST'])
@require_json(['image'])
def classification(payload):

  try:
    limit = int(payload.get('limit', 3))

  except ValueError as e:
    logging.error(e)
    raise BadRequest("Unable to process limit value '{}'".format(
        payload.get('limit')))

  try:
    threshold = float(payload.get('threshold', 0.0))

  except ValueError as e:
    logging.error(e)
    raise BadRequest("Unable to process threshold value '{}'".format(
        payload.get('threshold')))

  try:
    weighted = payload.get('weighted', False)
    if not isinstance(weighted, bool):
      raise TypeError('Weighted is not a boolean')

  except TypeError as e:
    logging.error(e)
    raise BadRequest("Unable to process weighted value '{}'".format(
        payload.get('weighted')))

  # Create an image object from the image data and validate it
  try:
    print('unpacking image info')
    image = payload.get('image')
    imagedata = re.sub('^data:image/.+;base64,', '', image)
    imagebytes = BytesIO(base64.b64decode(imagedata))

    image = Image.open(imagebytes)

  except base64.binascii.Error as e:
    logging.error(e)
    raise BadRequest('Unable to process image data: {}'.format(e))

  except TypeError as e:
    logging.error(e)
    message = e.message.message if isinstance(
        e.message, base64.binascii.Error) else e.message
    raise BadRequest('Unable to process image data: {}'.format(e.message))

  except IOError as e:
    logging.error(e)
    message = list(e.args)[0]
    raise BadRequest(message)

  if not isinstance(image, PngImageFile):
    raise BadRequest('Only png images are accepted')

  try:

    client_options = {'api_endpoint': f'{AI_REGION}-aiplatform.googleapis.com'}
    prediction_client = aiplatform.gapic.PredictionServiceClient(
        client_options=client_options)

    encoded_content = base64.b64encode(imagebytes.getvalue()).decode('utf-8')

    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content).to_value()
    instances = [instance]

    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.5,
        max_predictions=5,
    ).to_value()

    endpoint = prediction_client.endpoint_path(project=AI_PROJECT,
                                               location=AI_REGION,
                                               endpoint=AI_ENDPOINT_ID)

    response = prediction_client.predict(endpoint=endpoint,
                                         instances=instances,
                                         parameters=parameters)

    logging.info(response)
  except Exception as e:
    print('Error happened in inference')
    logging.error(e)
    raise InternalServerError('Error communicating with AutoML')

  predictions = []

  for prediction in response.predictions:

    glyph = prediction['displayNames'][0]
    confidence = prediction['confidences'][0]
    score = confidence * weights.get(glyph, 1) if weighted else confidence
    # Score might have been reduced due to weighting, so need to
    # check it again here
    if glyph != '--other--' and score >= threshold:
      predictions.append({'glyph': glyph, 'score': score})

  predictions = sorted(predictions, key=lambda p: p.get('score'), reverse=True)

  if limit < len(predictions):
    predictions = predictions[0:limit]

  return jsonify(code=200, success=True, result=predictions)
