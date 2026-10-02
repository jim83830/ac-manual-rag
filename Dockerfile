FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

# 先只複製 requirements，程式碼改動時才不用重裝套件（善用 layer cache）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["python", "app.py"]
