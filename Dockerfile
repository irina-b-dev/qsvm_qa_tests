# Use a base Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install Git and other necessary system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt requirements.txt

# Install the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code 
COPY . .

CMD ["python", "qsvm.py"]
