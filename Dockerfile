FROM tensorflow/tensorflow:2.14.0-gpu

RUN apt-get update && \
    # apt-get upgrade -y && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0

RUN python3 -m pip install --upgrade pip

COPY SeqNet2/requirements-pip.txt .

RUN python3 -m pip install --upgrade -r requirements-pip.txt

# Create a directory in the container
RUN mkdir /SeqNet

# Set the working directory
WORKDIR /SeqNet

# Define the volume
VOLUME /SeqNet

CMD ["./predict.sh"]
