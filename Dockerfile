FROM pytorch/pytorch
WORKDIR /app
RUN mkdir -p /Users/kahingleung/PycharmProjects/mylightning/
ADD trainer/lstm_stock.py /app
RUN conda install -c conda-forge pytorch-lightning
RUN pip install "ray[tune]"
RUN pip install yfinance
RUN pip install scikit-learn
RUN pip install matplotlib
CMD [ "python", "./lstm_stock.py" ]
