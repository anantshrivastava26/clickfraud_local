# API image. The model artifacts are baked in, so retraining means rebuilding the image.
FROM python:3.12-slim

WORKDIR /app
# No .pyc files in the image; unbuffered output so logs show up immediately in `docker logs`. PORT
# is the default listen port and can be overridden when the container starts (see CMD).
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000

# Dependencies first: this layer stays cached until requirements.txt changes, so code edits rebuild
# quickly.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the API needs at run time. train.py is included so the model can be retrained inside the
# container; make_samples.py and data/ are not.
COPY app ./app
COPY artifacts ./artifacts
COPY train.py .

EXPOSE 8000
# Shell form so ${PORT} is expanded when the container starts (hosts such as Railway or Render
# inject their own PORT). No --reload: this is the production command.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
