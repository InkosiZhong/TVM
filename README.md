# TVM: A Tile-based Video Management Framework

This is the official project page for paper *"TVM: A Tile-based Video Management Framework"*.
## Install
<details><summary>Expand to show installation details.</summary>
<p>

### 0 Download TVM

```bash
git clone https://github.com/InkosiZhong/TVM
```

### 1 NVIDIA GPU Driver & CUDA

Make sure your driver verision >= 520.56.06, and install a CUDA 11.3.

If you are using consumer-grade GPUs, use [nvidia-patch](https://github.com/keylase/nvidia-patch) to remove restriction on maximum number of simultaneous NVENC video encoding sessions.

### 2 Video Processing Framework

#### 2.1 Build FFmpeg from Source

```bash
sudo apt-get install yasm # nasm is also available
git clone https://github.com/FFmpeg/FFmpeg.git
cd FFmpeg
# for cuda support
git clone http://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
make
sudo make install
cd ..
mkdir -p $(pwd)/build_x64_release_shared 
./configure \
 --prefix=$(pwd)/build_x64_release_shared \
 --disable-static \
 --disable-stripping \
 --disable-doc \
 --enable-shared \
 --enable-cuda \
 --enable-cuvid \
 --enable-nvenc \
 --enable-nonfree \
 --enable-libnpp \
 --extra-cflags=-I/usr/local/cuda/include \
 --extra-ldflags=-L/usr/local/cuda/lib64
make -j -s && make install

# if you wanna use bin/ffmpeg
sudo vim /etc/ld.so.conf # $(pwd)/build_x64_release_shared/lib
sudo ldconfig

cd ..
```

#### 2.2 NVIDIA Video Codec SDK 12

Download the [NVIDIA Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk/download) into your workspace.

The version we use is `12.0.16`, we have not tested the following process on other versions, please choose the same version if possible.

#### 2.3 PyTorch Support

Install `torch==1.10.1` using pip or anaconda.

```bash
# anaconda
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
# pip
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```

#### 2.4 Build from Source

```bash
mkdir -p VideoProcessingFramework/install
mkdir -p VideoProcessingFramework/build
export ROOT=$(pwd)
export PATH_TO_SDK=$ROOT/Video_Codec_SDK_12.0.16
export SDK_INCLUDE=$PATH_TO_SDK/Interface
export SDK_LIB=$PATH_TO_SDK/Lib/linux/stubs/x86_64

export PATH_TO_FFMPEG=$ROOT/FFmpeg/build_x64_release_shared
export FFMPEG_INCLUD=$PATH_TO_FFMPEG/include 
export FFMPEG_LIB=$PATH_TO_FFMPEG/lib

export INSTALL_PREFIX=$ROOT/VideoProcessingFramework/install
export CUDACXX=$CUDA_HOME/bin/nvcc
export PKG_CONFIG_PATH=$FFMPEG_LIB/pkgconfig
export C_INCLUDE_PATH=$CUDA_HOME/include/:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include/:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64/:$LIBRARY_PATH
cd VideoProcessingFramework/build
cmake .. \
	-DGENERATE_PYTHON_BINDINGS:BOOL="1" \
	-DGENERATE_PYTORCH_EXTENSION:BOOL="1" \
	-DCMAKE_INSTALL_PREFIX:PATH="$INSTALL_PREFIX" \
	-DFFMPEG_DIR:PATH="$PATH_TO_FFMPEG" \
	-DAVFORMAT_INCLUDE_DIR:PATH="$FFMPEG_INCLUD" \
	-DAVUTIL_INCLUDE_DIR:PATH="$FFMPEG_INCLUD" \
	-DAVCODEC_INCLUDE_DIR:PATH="$FFMPEG_INCLUD" \
	-DAVCODEC_LIBRARY:PATH="$FFMPEG_LIB/libavcodec.so" \
	-DAVFORMAT_LIBRARY:PATH="$FFMPEG_LIB/libavformat.so" \
	-DAVUTIL_LIBRARY:PATH="$FFMPEG_LIB/libavutil.so" \
	-DVIDEO_CODEC_SDK_DIR:PATH="$PATH_TO_SDK" \
	-DVIDEO_CODEC_SDK_INCLUDE_DIR:PATH="$SDK_INCLUDE" \
	-DNVCUVID_LIBRARY:PATH="$SDK_LIB/libnvcuvid.so" \
	-DNVENCODE_LIBRARY:PATH="$SDK_LIB/libnvidia-encode.so"

make -j8
make install
```

After the compilation, the following files will be generated under the path `install/bin`.

```bash
libTC_CORE.so 
libTC.so 
PyNvCodec.cpython-38-x86_64-linux-gnu.so 
PytorchNvCodec.cpython-38-x86_64-linux-gnu.so
```

**Copy all these dynamic libraries into the root of TVM project.**

### 3 PyBGS (BGSLibrary)

#### 3.1 Build OpenCV from Source

```bash
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip
mkdir -p build && cd build

# configure
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x

# build & install
cmake --build . -j8
sudo make install

export OpenCV_DIR=~/OpenCV/build/
```

#### 3.2 Build from Source

```bash
git clone --recursive https://github.com/andrewssobral/bgslibrary.git
cd bgslibrary
python setup.py build
python setup.py install
pip install .
```

### 4 Python Libraries

Select the corresponding [pycuda](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pycuda) version according to the Cuda version and python version.

```bash
pip install -r requirements.txt
# install the query frameworks
cd query/blazeit
pip install -e .
cd ../supg
pip install -e .
```
</p>
</details>

## Prepare data

Thanks to the contribution of Daniel *et al.*, the datasets can be downloaded from [here](https://drive.google.com/drive/folders/1xRkmmOtyw7K3VNDRSP4-64ubNLe1K6Xi).

First, we need to initialize TVM:

```bash
mkdir -p cache/codec-test
mkdir -p logs/codec
python test/codec_test.py -i ../datasets/archie/2018-04-09 -o cache/codec-test -l logs/codec -c config/archie.json
python scripts/system_init.py -l logs/codec -t codec

mkdir -p logs/dnn
python test/nn_test.py -l logs/dnn
python scripts/system_init.py -l logs/dnn -t dnn -m 1024
```

To create hierarchical tile layouts and tiled video data, run the following command:

```bash
mkdir -p cache/tiling
python scripts/roi_tiling.py -i ../datasets/archie/2018-04-09 -o ../datasets/archie/2018-04-09-tile -c config/archie.json 
python scripts/tiling_index.py -i ../datasets/archie/2018-04-09-tile -o query/examples/archie/cache/tiling -c config/archie.json 
python scripts/roi_tiling.py -i ../datasets/archie/2018-04-10 -o ../datasets/archie/2018-04-10-tile -c config/archie.json -t cache/tiling
```

Generate the semantic embeddings for each ROI-based tiled frame:

```bash
python query/examples/archie/gen_embed.py
```

> **NOTE**: Modify the directory address in the files to match the actual file location before execution.

## Evaluate Queries

The script will perform aggregation query, limit query, approximate select query and track query.

```bash
python query/examples/archie/eval.py
```

> **NOTE**: Modify the directory address in the files to match the actual file location before execution.
