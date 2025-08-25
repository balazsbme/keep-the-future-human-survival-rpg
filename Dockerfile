# Use the official Python image as a base
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy the rest of your application code
COPY . .

# Expose the port that your Flask app will listen on
# Cloud Run expects your application to listen on the port specified by the PORT environment variable.
ENV PORT 8080
EXPOSE ${PORT}

# Run the application using Gunicorn (a production-ready WSGI HTTP Server)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "rag_service:app"]
