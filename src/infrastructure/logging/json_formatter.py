from __future__ import annotations

from datetime import datetime
import json

import json_logging.formatters


class JsonFormatter(json_logging.formatters.JSONLogFormatter):
    def format(self, record):
        json_record = {
            'time': datetime.utcfromtimestamp(record.created).isoformat(
                sep=' ', timespec='milliseconds'
            ),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        if record.exc_info or record.exc_text:
            json_record['exception'] = {
                'asString': self.format_exception(record.exc_info)
                if record.exc_info
                else record.exc_text,
            }

        return json.dumps(json_record)
