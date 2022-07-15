service: classification-api2
runtime: python38

default_expiration: "10m"

entrypoint: gunicorn -b :$PORT --log-level INFO --reload main:app

handlers:
  - url: /.*
    script: auto
    secure: always

env_variables:
  AI_PROJECT: {{AI_PROJECT}}
  AI_REGION: {{AI_REGION}}
  AI_ENDPOINT: {{AI_ENDPOINT}}

inbound_services:
  - warmup

automatic_scaling:
  min_instances: {{MIN_INSTANCES}}
