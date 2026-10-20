FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk:26.1.0-9.1.0-torch_npu2.7.1.post6-a3-openeuler24.03-py3.10

USER root
SHELL ["/bin/bash", "-lc"]

ENV APOLLO_ROOT=/opt/apollo
ENV APOLLO_ENV=/opt/apollo/env
ENV APOLLO_WORKSPACE=/opt/apollo/workspace

RUN mkdir -p "${APOLLO_ENV}" "${APOLLO_WORKSPACE}"

# conda-pack生成的已验证Python环境；ADD会自动解压tar.gz。
ADD docker/context/env2_apollo.tar.gz /opt/apollo/env/

# 修复Conda环境中原有的绝对路径。
RUN /opt/apollo/env/bin/conda-unpack

# 复制已经固定版本的全部代码和预训练权重。
COPY Apollo-Vision-Net /opt/apollo/workspace/Apollo-Vision-Net
COPY DrivingSDK /opt/apollo/workspace/DrivingSDK
COPY mmcv /opt/apollo/workspace/mmcv
COPY mmdetection_2.19 /opt/apollo/workspace/mmdetection_2.19
COPY mmsegmentation_0.20 /opt/apollo/workspace/mmsegmentation_0.20
COPY mmdetection3d_0.18.1 /opt/apollo/workspace/mmdetection3d_0.18.1
COPY COMPONENT_VERSIONS.md WEIGHTS.sha256 /opt/apollo/workspace/

# 恢复当前已验证的MMCV aarch64/Python 3.10动态库。
COPY docker/artifacts/_ext.cpython-310-aarch64-linux-gnu.so \
     /opt/apollo/workspace/mmcv/mmcv/_ext.cpython-310-aarch64-linux-gnu.so

COPY docker/entrypoint.sh /opt/apollo/entrypoint.sh
RUN chmod 755 /opt/apollo/entrypoint.sh

ENV PATH=/opt/apollo/env/bin:${PATH}
ENV PYTHONPATH=/opt/apollo/workspace/mmcv:/opt/apollo/workspace/mmdetection_2.19:/opt/apollo/workspace/mmsegmentation_0.20:/opt/apollo/workspace/mmdetection3d_0.18.1:/opt/apollo/workspace/Apollo-Vision-Net:${PYTHONPATH}


WORKDIR /opt/apollo/workspace/Apollo-Vision-Net

ENTRYPOINT ["/opt/apollo/entrypoint.sh"]
CMD ["bash"]
