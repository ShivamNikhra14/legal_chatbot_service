# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
ENV PORT 8080

# Create and set the working directory
WORKDIR $APP_HOME

# Install system dependencies (if any, e.g., for certain libraries)
# RUN apt-get update && apt-get install -y ...

# Install Python dependencies
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY main.py .

# Expose the port the app runs on
EXPOSE $PORT

# Define the command to run your application
# Uvicorn is the server that runs your FastAPI app.
# We listen on 0.0.0.0 to accept connections from any IP.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]