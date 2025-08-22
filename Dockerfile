FROM python:3.12-slim 

WORKDIR /app

# Copy from current location to app folder
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Run the main.py in python3
CMD ["streamlit", "run", "main.py"]

