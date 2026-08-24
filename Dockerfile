# Multi-Architecture IR Analysis 
# - Environment only contains free/open-source backend tools + their Python dependencies.
# - Code and corpus are not included in the image, both can be obtained by the Github Repo (corpus link is in README.md file)
#
#   docker build -t multi-arch-ir-analysis .

FROM ubuntu:24.04

ARG GHIDRA_VERSION=12.1
ARG GHIDRA_BUILD_DATE=20260513
ARG RETDEC_VERSION=v5.0
ARG R2_VERSION=6.2.0

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    GHIDRA_INSTALL_DIR=/opt/ghidra \
    RETDEC_BIN=/opt/retdec/retdec-decompiler \
    PATH=/opt/venv/bin:/opt/retdec:${PATH}

# --- OS packages + Temurin 21 JDK ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg wget unzip xz-utils \
        python3.12 python3.12-venv python3-pip \
        build-essential \
    && wget -qO- https://packages.adoptium.net/artifactory/api/gpg/key/public \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/adoptium.gpg \
    && echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print $2}' /etc/os-release) main" \
        > /etc/apt/sources.list.d/adoptium.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        temurin-21-jdk \
    && rm -rf /var/lib/apt/lists/*

# --- radare2 ---
RUN curl -fsSL -o /tmp/radare2.deb \
        "https://github.com/radareorg/radare2/releases/download/${R2_VERSION}/radare2_${R2_VERSION}_amd64.deb" \
    && apt-get update && apt-get install -y --no-install-recommends /tmp/radare2.deb \
    && rm -rf /tmp/radare2.deb /var/lib/apt/lists/*

# --- Ghidra  ---
RUN curl -fsSL -o /tmp/ghidra.zip \
        "https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_${GHIDRA_VERSION}_build/ghidra_${GHIDRA_VERSION}_PUBLIC_${GHIDRA_BUILD_DATE}.zip" \
    && unzip -q /tmp/ghidra.zip -d /opt \
    && mv /opt/ghidra_${GHIDRA_VERSION}_PUBLIC "${GHIDRA_INSTALL_DIR}" \
    && chmod +x "${GHIDRA_INSTALL_DIR}"/support/*.sh \
    && rm /tmp/ghidra.zip

# --- RetDec ---
RUN curl -fsSL -o /tmp/retdec.tar.xz \
        "https://github.com/avast/retdec/releases/download/${RETDEC_VERSION}/RetDec-${RETDEC_VERSION}-Linux-Release.tar.xz" \
    && mkdir -p /opt/retdec \
    && tar -xJf /tmp/retdec.tar.xz -C /opt/retdec --strip-components=1 \
    && rm /tmp/retdec.tar.xz

# --- Python dependencies ---
RUN python3.12 -m venv /opt/venv
WORKDIR /workspace
COPY requirements.txt .
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

CMD ["bash"]
