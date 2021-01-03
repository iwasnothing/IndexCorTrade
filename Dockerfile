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
ENTRYPOINT [ "python", "trainer/lstm_stock.py" ]
