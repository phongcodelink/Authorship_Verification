# Use the python:3.10 as the base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set environment variables
ENV PATH "/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE True
ENV PYTHONPATH app
ENV PYTHONUNBUFFERED True

# Create a virtual environment and install dependencies
RUN python -m venv /opt/venv
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip --requirement requirements.txt
RUN python -m nltk.downloader averaged_perceptron_tagger -d /opt/venv/nltk_data

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . app

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:80", "--timeout", "600"]
