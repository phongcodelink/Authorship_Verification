# Use the nvidia pytorch image as the base
FROM nvcr.io/nvidia/pytorch:23.08-py3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE True
ENV PYTHONPATH app
ENV PYTHONUNBUFFERED True

# Install dependencies directly without using virtual environment
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install --requirement requirements.txt
RUN python -m nltk.downloader averaged_perceptron_tagger

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . app

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:80", "--timeout", "600"]
