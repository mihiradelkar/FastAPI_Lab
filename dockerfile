# Use an official Python runtime as a parent image
FROM python:3.9
# FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy model directory (trained models)
COPY model/ ./model/

# Copy source code
COPY src/ ./src/

# Expose port 8000 for FastAPI
EXPOSE 8000

# Change to src directory
WORKDIR /app/src

# Run the script when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]