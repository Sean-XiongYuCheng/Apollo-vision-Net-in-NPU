src_path=$(pwd)

SYSTEM=openeular
grep -q "openEuler" /etc/os-release && SYSTEM=openeular || SYSTEM=ubuntu
echo "系统类别: $SYSTEM"

cd $src_path
#安装依赖
echo "模型依赖开始安装"
torch_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
if echo "$torch_version" | grep -q "2.1.0"; then
    pip install -r requirements_pytorch2.1.0.txt
else
    pip install -r requirements_pytorch2.7.1.txt
fi

##安装mmcv
echo "mmcv开始安装"
git clone -b 1.x https://github.com/open-mmlab/mmcv
cp mmcv_config.patch mmcv/
cd mmcv
git apply --reject --whitespace=fix mmcv_config.patch
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
cd ..

##安装mmdet3d
echo "mmdet3d开始安装"
git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
cp mmdet3d_config.patch mmdetection3d/
cd mmdetection3d
git apply --reject --whitespace=fix mmdet3d_config.patch
pip install -v -e . --no-build-isolation
cd ..

##模型代码更新
echo "模型代码开始更新"
git clone https://github.com/hustvl/MapTR.git
cp MapTR.patch MapTR
cd MapTR
git checkout 1b435fd9f0db9a14bb2a9baafb565200cc7028a2
git apply --reject --whitespace=fix MapTR.patch
cd ../

##安装tcmalloc(openeular或者ubuntu)
echo "tcmalloc开始安装"
if [ "$SYSTEM" == "openeular" ]; then
    mkdir gperftools
    cd gperftools
    wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz
    tar -zvxf gperftools-2.16.tar.gz
    cd gperftools-2.16
    ./configure --prefix=/usr/local/lib --with-tcmalloc-pagesize=64
    make
    make install
    echo '/usr/local/lib/lib/' >> /etc/ld.so.conf
    ldconfig
    export LD_LIBRARY_PATH=/usr/local/lib/lib/:$LD_LIBRARY_PATH
    export PATH=/usr/local/lib/bin:$PATH
    export LD_PRELOAD=/usr/local/lib/lib/libtcmalloc.so.4
else
    apt-get update
    apt -y install autoconf
    apt -y install libtool
    git clone https://github.com/libunwind/libunwind.git
    cd libunwind
    autoreconf -i
    ./configure --prefix=/usr/local
    make -j128
    make install
    cd ..
    wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz
    tar -xf gperftools-2.16.tar.gz && cd gperftools-2.16
    ./configure --prefix=/usr/local/lib --with-tcmalloc-pagesize=64
    make -j128
    make install
    export LD_PRELOAD="$LD_PRELOAD:/usr/local/lib/lib/libtcmalloc.so"
fi

cd $src_path
