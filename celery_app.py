from celery import Celery
from celery.schedules import crontab

import sys
import os

# Добавляем корень проекта в PYTHONPATH
sys.path.append(os.path.dirname(__file__))


app = Celery(
    "project",
    broker="redis://localhost:6379/0",   # или RabbitMQ
    backend="redis://localhost:6379/1",
    include=["app.tasks"],  # Explicitly include tasks module
)

# Explicitly import tasks to ensure they're registered
app.autodiscover_tasks(["app"], force=True)

# Configure Celery Beat schedule
app.conf.beat_schedule = {
    "check-rss-feed": {
        "task": "app.tasks.check_rss_feed",
        # "schedule": crontab(minute='*/30'),  # Every 30 minutes
        # Alternative schedules:
        # "schedule": crontab(hour='*/1'),  # Every hour
        # "schedule": crontab(minute=0, hour='*/6'),  # Every 6 hours
        "schedule": 10.0,  # Every 10 seconds
    },
}
app.conf.timezone = "UTC"
