# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your requirements file first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your code and data into the container
COPY . .

# Hugging Face Spaces strictly require applications to run on port 7860
EXPOSE 7860

# Command to run the FastAPI server on the correct port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]