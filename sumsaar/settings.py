import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIRS = {
    'runtime': os.path.join(PROJECT_DIR, 'runtime')
}

OLLAMA_URL = "http://srsw.cdot.in:11434"
CRAWLER_URL = "http://192.168.3.82:11235/"