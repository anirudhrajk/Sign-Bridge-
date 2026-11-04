# SignBridge web app. Models and the sign library are prebuilt in static/, so no ML libraries are needed.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Large browser libraries are fetched here (checked against npm hashes) instead of stored in git.
RUN python scripts/fetch_vendor.py
ENV PORT=8000 HOST=0.0.0.0 TRUST_PROXY=1 HTTPS=1 DEMO_ACCOUNTS=0
EXPOSE 8000
CMD ["python", "serve.py"]
