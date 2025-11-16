FROM python:3.11-slim

WORKDIR /opt

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Clone the master branch of the repository 
RUN git clone --depth 1 --branch main https://github.com/balazsbme/keep-the-future-human-survival-rpg.git app

WORKDIR /opt/app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default port used by Hugging Face Spaces
ENV PORT=7860
EXPOSE 7860

# Launch the web server
CMD ["python", "web_service.py"]