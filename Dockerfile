FROM python:3.11-slim

WORKDIR /opt

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY . app

WORKDIR /opt/app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default port used by Hugging Face Spaces
ENV PORT=7860
EXPOSE 7860

# Launch the web server
CMD ["python", "web_service.py"]