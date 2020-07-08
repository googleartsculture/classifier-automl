# A dockerfile purely to speed up builds
# grpcio otherwise takes a long time to load up.
FROM python:3.7

# COPY Dependencies
COPY src/requirements.txt ./

RUN pip3 install -r requirements.txt

ENTRYPOINT [ "bash" ]
