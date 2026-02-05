
# # Force x86_64 base image (TF 1.15 CPU)
# FROM --platform=linux/amd64 tensorflow/tensorflow:1.15.5-py3

# WORKDIR /app
# COPY . /app

# # System packages for dlib, ffmpeg, etc.
# RUN apt-get update && apt-get install -y \
#     build-essential cmake ffmpeg libsm6 libxext6 libgl1-mesa-glx \
#     && rm -rf /var/lib/apt/lists/*

# # Install compatible Python dependencies
# RUN pip install --upgrade pip && \
#     pip install \
#       numpy==1.18.5 \
#       scipy==1.4.1 \
#       scikit-learn \
#       matplotlib==3.2.2 \
#       pillow==7.0.0 \
#       h5py==2.10.0 \
#       keras==2.2.4 \
#       tensorflow==1.15.5 \
#       editdistance==0.3.1 \
#       python-dateutil==2.6.0 \
#       nltk==3.2.2 \
#       sk-video==1.1.7 \
#       theano==0.9.0 \
#       dlib \
#       moviepy && \
#     pip install -e ./LipType

# CMD ["/bin/bash"]


FROM --platform=linux/amd64 tensorflow/tensorflow:1.15.5-py3

WORKDIR /app

COPY . /app

# --- Add these lines ---
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ffmpeg \
    python3-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    pkg-config
# ------------------------

RUN pip install --upgrade pip && \
    pip install \
      numpy==1.18.5 \
      scipy==1.4.1 \
      scikit-learn \
      matplotlib==3.2.2 \
      pillow==7.0.0 \
      h5py==2.10.0 \
      keras==2.2.4 \
      tensorflow==1.15.5 \
      editdistance==0.3.1 \
      python-dateutil==2.6.0 \
      nltk==3.2.2 \
      sk-video==1.1.7 \
      theano==0.9.0 \
      moviepy && \
    pip install -e ./LipType
