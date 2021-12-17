####
# This Dockerfile is used in order to build a container that runs the Spring Boot application
#
# Build the image with:
#
# docker build -f docker/Dockerfile -t python/sample-basic .
#
# Then run the container using:
#
# docker run -i --rm -p 8081:8081 python/sample-basic
####
FROM python:3.8

WORKDIR /projects

RUN python3 -m venv venv
RUN . venv/bin/activate

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN /usr/local/bin/python -m pip install --upgrade pip

# optimize image caching
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8081
CMD [ "waitress-serve", "--port=8081", "app:app"]

