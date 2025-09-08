FROM python:3.12.3-slim
EXPOSE 8000

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements_docker.txt .
RUN python -m pip install -r requirements_docker.txt

WORKDIR /app
COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
