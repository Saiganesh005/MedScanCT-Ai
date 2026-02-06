# Use Python base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Streamlit uses 8501, FastAPI uses 8000)
EXPOSE 8501

# Run Streamlit app (change to app.py if you want FastAPI instead)
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
