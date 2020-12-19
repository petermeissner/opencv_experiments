# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.148.1/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version: 3, 3.9, 3.8, 3.7, 3.6
ARG VARIANT="3"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# [Option] Install Node.js
ARG INSTALL_NODE="true"
ARG NODE_VERSION="lts/*"
RUN if [ "${INSTALL_NODE}" = "true" ]; then su vscode -c "source /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
# COPY requirements.txt /tmp/pip-tmp/
# RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
#    && rm -rf /tmp/pip-tmp




# Install minimal prerequisites (Ubuntu 18.04 as reference)
RUN apt-get update \
    && sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        g++  \
    && rm -rf /var/lib/apt/lists/*


# change workdir
RUN mkdir /opencv
WORKDIR /opencv


# Download and unpack sources
RUN wget -q -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
RUN wget -q -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/master.zip
RUN unzip -q opencv.zip
RUN unzip -q opencv_contrib.zip

# Create build directory and switch into it
RUN mkdir -p /opencv/build && cd build

WORKDIR /opencv/build

# Configure
RUN cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules ../opencv-master

# Build
RUN cmake --build .


# RUN pip3 --disable-pip-version-check --no-cache-dir install ipykernel

RUN pip3 --disable-pip-version-check --no-cache-dir install \
argon2-cffi==20.1.0 \
async-generator==1.10 \
attrs==20.3.0 \
backcall==0.2.0 \
bleach==3.2.1 \
cffi==1.14.4 \
cycler==0.10.0 \
decorator==4.4.2 \
defusedxml==0.6.0 \
entrypoints==0.3 \
ipykernel==5.3.4 \
ipython==7.19.0 \
ipython-genutils==0.2.0 \
ipywidgets==7.5.1 \
jedi==0.17.2 \
Jinja2==2.11.2 \
jsonschema==3.2.0 \
jupyter==1.0.0 \
jupyter-client==6.1.7 \
jupyter-console==6.2.0 \
jupyter-core==4.7.0 \
jupyterlab-pygments==0.1.2 \
kiwisolver==1.3.1 \
MarkupSafe==1.1.1 \
matplotlib==3.3.3 \
mistune==0.8.4 \
nbclient==0.5.1 \
nbconvert==6.0.7 \
nbformat==5.0.8 \
nest-asyncio==1.4.3 \
notebook==6.1.5 \
numpy==1.19.4 \
opencv-contrib-python==4.4.0.46 \
opencv-python==4.4.0.46 \
packaging==20.7 \
pandocfilters==1.4.3 \
parso==0.7.1 \
pexpect==4.8.0 \
pickleshare==0.7.5 \
Pillow==8.0.1 \
prometheus-client==0.9.0 \
prompt-toolkit==3.0.8 \
ptyprocess==0.6.0 \
pycparser==2.20 \
Pygments==2.7.2 \
pyparsing==2.4.7 \
pyrsistent==0.17.3 \
python-dateutil==2.8.1 \
pyzmq==20.0.0 \
qtconsole==5.0.1 \
QtPy==1.9.0 \
Send2Trash==1.5.0 \
six==1.15.0 \
terminado==0.9.1 \
testpath==0.4.4 \
tornado==6.1 \
traitlets==5.0.5 \
wcwidth==0.2.5 \
webencodings==0.5.1 \
widgetsnbextension==3.5.1

    
# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# [Optional] Uncomment this line to install global node packages.
# RUN su vscode -c "source /usr/local/share/nvm/nvm.sh && npm install -g <your-package-here>" 2>&1

 RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get -y install --no-install-recommends libopencv-dev
     
RUN mkdir /darknet
WORKDIR /darknet
RUN git clone https://github.com/pjreddie/darknet.git
WORKDIR /darknet/darknet
RUN sed -i 's/OPENCV=0/OPENCV=1/g' Makefile
RUN make

RUN wget https://pjreddie.com/media/files/yolov3.weights


