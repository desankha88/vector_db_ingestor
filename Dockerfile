FROM python:latest
WORKDIR /app
COPY requirements/requirements.txt /app/
RUN pip install -r requirements.txt
COPY . /app/
# Corrected path for the main application file
CMD ["python", "src/PdfToChromaClientApp.py"]