FROM python:3.9-slim
COPY . /app
WORKDIR /app
COPY . .
#ENV PYTHONUNBUFFERED=1
RUN pip install -r requirements.txt
EXPOSE 5000
CMD [ "python", "./app.py" ]