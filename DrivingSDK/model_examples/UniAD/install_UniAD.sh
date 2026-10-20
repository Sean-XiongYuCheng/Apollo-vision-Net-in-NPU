#!/bin/bash
set -e
src_path=$(pwd)

SYSTEM=ubuntu
grep -q "openEuler" /etc/os-release && SYSTEM=openeular || SYSTEM=ubuntu
echo "系统类别: $SYSTEM"

#安装 python 依赖（按 torch 版本区分）
echo "模型依赖开始安装"
pip install -r requirements_torch2.7.1.txt

# 2. 安装 mmcv
echo "mmcv开始安装"
git clone -b 1.x https://github.com/open-mmlab/mmcv
cp -f mmcv.patch mmcv/
cd mmcv
git apply --reject --whitespace=fix mmcv.patch
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
cd ..

# 3. 安装 mmdet3d
echo "mmdet3d开始安装"
git clone -b v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git
cp -f mmdet3d.patch mmdetection3d/
cd mmdetection3d
git apply --reject --whitespace=fix mmdet3d.patch
pip install -v -e . --no-build-isolation
cd ..

# 4. 拉取 UniAD 源码
echo "UniAD源码开始拉取"
git clone https://github.com/OpenDriveLab/UniAD.git
cd UniAD
git checkout 7b5bf15e0e49522b6553ddc48e67833e8f5f0f52
cd ..

# 5. 安装 tcmalloc（性能优化，可选）
echo "tcmalloc开始安装"
if ldconfig -p 2>/dev/null | grep -q "libtcmalloc.so"; then
    echo "检测到系统已安装 tcmalloc，跳过源码安装"
    cd "$src_path"
    echo "============================================================"
    echo "  UniAD 环境安装完成"
    echo "  详细说明见 README.md"
    echo "============================================================"
    exit 0
fi
if [ "$SYSTEM" == "openeular" ]; then
    mkdir -p gperftools
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
    if [ ! -d libunwind ]; then
        git clone https://github.com/libunwind/libunwind.git
    fi
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

cd "$src_path"
echo "============================================================"
echo "  UniAD 环境安装完成"
echo "  详细说明见 README.md"
echo "============================================================"
