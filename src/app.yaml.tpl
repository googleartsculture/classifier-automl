service: classification-api2
runtime: python37

default_expiration: "10m"

entrypoint: gunicorn -b :$PORT --log-level INFO --reload main:app

handlers:
  - url: /.*
    script: auto
    secure: always

env_variables:
  AUTO_ML_PROJECT: {{AUTO_ML_PROJECT}}
  AUTO_ML_LOCATION: {{AUTO_ML_LOCATION}}
  AUTO_ML_MODEL: {{AUTO_ML_MODEL}}

inbound_services:
  - warmup

automatic_scaling:
  min_instances: {{MIN_INSTANCES}}
