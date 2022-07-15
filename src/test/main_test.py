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

# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
# pylint: disable=invalid-name
# pylint: disable=missing-class-docstring
# pylint: disable=consider-using-with
# pylint: disable=super-with-arguments
# pylint: disable=unspecified-encoding
"""
Main test module
"""

import os
import json
from test.utils import TestbedTestCase
from unittest.mock import patch

import main

# with open('test/valid_png.txt', 'r') as png:
#     VALID_PNG_IMAGE_DATA = png.read().replace('\n', '')

DIR_PATH = os.path.dirname(__file__)
VALID_JPG_IMAGE_DATA = open(os.path.join(DIR_PATH, 'valid_jpg.txt'), 'r')
JPG_IMAGE_STR = VALID_JPG_IMAGE_DATA.read().replace('\n', '')
VALID_PNG_IMAGE_DATA = open(os.path.join(DIR_PATH, 'valid_png.txt'), 'r')
PNG_IMAGE_STR = VALID_PNG_IMAGE_DATA.read().replace('\n', '')


class MockClassification():

  def __init__(self, score=0):
    """ init function """
    self.score = score


class MockPrediction():
  annotationSpecId = 'someId'
  display_name = 'a1'
  classification = MockClassification(score=0.92)


class MockPredictResponse():

  predictions = [MockPrediction()]
  processedInput = ''
  metadata = None
  deployed_model_id = '2423439287'


class MainTests(TestbedTestCase):

  def setUp(self):
    super().setUp()
    self.app = main.app
    self.app.config['TESTING'] = True
    self.client = self.app.test_client()

  def test_request_to_unknown_url(self):
    response = self.client.get('/unknown')
    self.assertEqual('404 NOT FOUND', response.status)
    data = json.loads(response.data)
    self.assertFalse(data.get('success'))
    self.assertEqual(404, data.get('code'))
    # We don't really care what the message says,
    # as long as it was a 404 status

  def test_classification_request_with_incorrect_http_method(self):
    response = self.client.get('/classification')
    self.assertEqual('405 METHOD NOT ALLOWED', response.status)
    data = json.loads(response.data)
    self.assertFalse(data.get('success'))
    self.assertEqual(405, data.get('code'))
    # We don't really care what the message says,
    # as long as it was a 405 status

  def test_classification_request_with_missing_json_header(self):
    response = self.client.post('/classification', data={})
    self.assertEqual('400 BAD REQUEST', response.status)
    data = json.loads(response.data)
    self.assertFalse(data.get('success'))
    self.assertEqual(400, data.get('code'))
    self.assertEqual('The request content type must be application/json',
                     data.get('message'))

  def test_classification_request_with_empty_payload(self):
    response = self.client.post('/classification', json={})
    self.assertEqual('400 BAD REQUEST', response.status)
    data = json.loads(response.data)
    self.assertFalse(data.get('success'))
    self.assertEqual(400, data.get('code'))
    self.assertEqual('The request is missing the image parameter',
                     data.get('message'))

  def test_classification_request_with_invalid_image_data(self):
    response = self.client.post('/classification',
                                json={'image': 'invaliddata'})
    self.assertEqual('400 BAD REQUEST', response.status)
    data = json.loads(response.data)
    self.assertFalse(data.get('success'))
    self.assertEqual(400, data.get('code'))
    self.assertTrue(
        data.get('message').startswith('Unable to process image data:'))

  def test_classification_request_with_non_png_image_data(self):
    response = self.client.post('/classification',
                                json={'image': JPG_IMAGE_STR})
    self.assertEqual('400 BAD REQUEST', response.status)
    data = json.loads(response.data)
    self.assertFalse(data.get('success'))
    self.assertEqual(400, data.get('code'))
    self.assertEqual('Only png images are accepted', data.get('message'))

  def test_classification_request_with_invalid_model_details(self):

    response = self.client.post('/classification',
                                json={
                                    'image': PNG_IMAGE_STR,
                                    'model_name': 'invalid_model',
                                    'model_version': 'invalid_version'
                                })
    data = json.loads(response.data)
    print(data)
    self.assertEqual('500 INTERNAL SERVER ERROR', response.status)

    self.assertFalse(data.get('success'))
    self.assertEqual(500, data.get('code'))
    self.assertEqual('Error communicating with AutoML', data.get('message'))

  def test_classification_request_with_invalid_limit(self):
    response = self.client.post('/classification',
                                json={
                                    'image': PNG_IMAGE_STR,
                                    'limit': 'invalid'
                                })
    self.assertEqual('400 BAD REQUEST', response.status)
    data = json.loads(response.data)
    self.assertFalse(data.get('success'))
    self.assertEqual(400, data.get('code'))
    self.assertEqual("Unable to process limit value 'invalid'",
                     data.get('message'))

  def test_classification_request_with_invalid_threshold(self):
    response = self.client.post('/classification',
                                json={
                                    'image': PNG_IMAGE_STR,
                                    'threshold': 'invalid'
                                })
    self.assertEqual('400 BAD REQUEST', response.status)
    data = json.loads(response.data)
    self.assertFalse(data.get('success'))
    self.assertEqual(400, data.get('code'))
    self.assertEqual("Unable to process threshold value 'invalid'",
                     data.get('message'))

  def test_classification_request_with_invalid_weighted_flag(self):
    response = self.client.post('/classification',
                                json={
                                    'image': PNG_IMAGE_STR,
                                    'weighted': 'invalid'
                                })
    self.assertEqual('400 BAD REQUEST', response.status)
    data = json.loads(response.data)
    self.assertFalse(data.get('success'))
    self.assertEqual(400, data.get('code'))
    self.assertEqual("Unable to process weighted value 'invalid'",
                     data.get('message'))


# @TODO stub as this actually calls automl or move to integration testing

  @patch.object(main.aiplatform.gapic.PredictionServiceClient,
                'predict',
                return_value=MockPredictResponse())
  def test_valid_classification_requests(self, mock_method):

    response = self.client.post('/classification',
                                json={'image': PNG_IMAGE_STR})
    data = json.loads(response.data)
    self.assertEqual('200 OK', response.status)
    data = json.loads(response.data)
    self.assertTrue(data.get('success'))
    self.assertEqual(200, data.get('code'))
    self.assertIsInstance(data.get('result'), list)
    for item in data.get('result'):
      self.assertIsInstance(item, dict)
      self.assertIn('score', item)
      self.assertIsInstance(item.get('score'), float)
      self.assertIn('glyph', item)
      self.assertIsInstance(item.get('glyph'), str)

  def test_warmup_request_responds_200(self):
    """
        Asserts that a reuqest to /_ah/warmup is handled.
        """
    response = self.client.get('/_ah/warmup')
    self.assertEqual('200 OK', response.status)
