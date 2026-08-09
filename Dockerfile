# Base image
FROM python:3.12-slim

# Define the working directory
WORKDIR /appLogInsightAgent

# Installs system dependencies for compiling Python libraries
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Copy the dependency file
COPY requirements.txt .
# Install the dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Exposes the port used by Streamlit
EXPOSE 8501

# Command to start the application
CMD ["streamlit", "run", "App.py"]