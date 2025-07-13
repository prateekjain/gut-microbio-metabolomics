web: gunicorn app:server --timeout 60 --workers 2 --worker-class gthread --threads 4 --max-requests 1000 --max-requests-jitter 100 --preload
