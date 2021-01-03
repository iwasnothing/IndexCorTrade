FROM pytorch/pytorch
WORKDIR /app
RUN mkdir =p /app/trainer
RUN mkdir -p /Users/kahingleung/PycharmProjects/mylightning/
ADD trainer/lstm_stock.py /app/trainer/lstm_stock.py
RUN conda install -c conda-forge pytorch-lightning
RUN pip install "ray[tune]"
RUN pip install yfinance
RUN pip install scikit-learn
RUN pip install matplotlib
# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN apt-get install wget -y
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg
ENTRYPOINT [ "python", "trainer/lstm_stock.py" ]
