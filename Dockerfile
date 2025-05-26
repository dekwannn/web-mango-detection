# Gunakan image dasar dengan Python
FROM python:3.10-slim

# Install libGL dan dependencies lain yang dibutuhkan OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua kode ke image
COPY . /app
WORKDIR /app

# Jalankan app
CMD ["python", "app.py"]
