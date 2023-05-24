# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. 

# maskrcnn_benchmark and coco api dependencies
pip install ninja==1.10.2.3 yacs==0.1.8 cython==0.29.28 matplotlib==3.5.1 tqdm==4.63.0 opencv-python==4.5.5.64

# MIC dependencies
pip install timm==0.6.11 kornia==0.5.8 einops==0.4.1

export INSTALL_DIR=$PWD/MIC

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
find . -type f -name "*.py" -print0 | xargs -0 sed -i 's/np.float/float/g'
python setup.py build_ext install

# install cityscapesScripts
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
find . -type f -name "*.py" -print0 | xargs -0 sed -i 's/np.float/float/g'
python setup.py build_ext install

# install sa-da-faster
cd $INSTALL_DIR
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

# copy required file to cityscapesScripts toolkit
cp tools/cityscapes/instances2dict_with_polygons.py $INSTALL_DIR/cityscapesScripts/cityscapesscripts/evaluation

# recompile cityscapesScripts
cd cityscapesScripts/
python setup.py build_ext install

pip install h5py==3.6.0 scipy==1.8.0

cd $INSTALL_DIR
unset INSTALL_DIR
