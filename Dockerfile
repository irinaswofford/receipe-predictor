# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make sure that the model directory is recognized as a Python package
RUN touch /app/model/__init__.py

# Set the Python path to include the app directory
ENV PYTHONPATH=/app

# Set the Flask app environment variable
ENV FLASK_APP=app.main

# Train the model
RUN python -m model.train

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run Flask app when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]
