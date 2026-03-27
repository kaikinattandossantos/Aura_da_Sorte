#!/bin/bash
set -e

echo "Starting uvicorn..."
uvicorn Data.app.main:app --host 127.0.0.1 --port 8000 --workers 2 &

echo "Starting nginx..."
nginx -g "daemon off;"
