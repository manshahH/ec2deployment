FROM python:3.12

WORKDIR /app

COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/main.py main.py
COPY preprocessing_objects/ preprocessing_objects/
COPY trained_models_pkl/ trained_models_pkl/


EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
