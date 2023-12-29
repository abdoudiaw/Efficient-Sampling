# hash:sha256:c90c5ec2ae513f6db38dc687b754f50a233ff2d5e27c24c29f906ea5743147da
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    multiprocess==0.70.12.2 \
    mystic==0.3.8 \
    sklearn==0.0 \
    subprocess32==3.5.4

RUN pip3 install -U --no-cache-dir \
    torch==1.10.2

COPY postInstall /
RUN /postInstall
