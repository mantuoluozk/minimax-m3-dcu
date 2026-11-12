ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG BUILD_NATIVE=1
WORKDIR /opt/minimax-m3-dcu
ENV ROCM_HOME=/opt/dtk PYTORCH_ROCM_ARCH=gfx938 MAX_JOBS=2
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/dtk/.hyhal/rocm_smi/lib
COPY sglang-das/python/ sglang-das/python/
COPY native/ native/
COPY build_native.py LICENSE LICENSE-lightop ./
COPY P.sh D.sh router.sh ./
COPY deploy/ deploy/
RUN test ! -e /dev/kfd && test ! -e /dev/mkfd && \
    python3 deploy/prepare_decode.py && \
    if [ "$BUILD_NATIVE" = "1" ]; then python3 build_native.py; fi
ENTRYPOINT ["/bin/bash", "/opt/minimax-m3-dcu/deploy/entrypoint.sh"]
CMD ["help"]
