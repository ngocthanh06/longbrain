"""Standalone test tree for the Google Drive connector.

Deliberately NOT wired into llamaindex-service/tests: this connector's
dependencies (google-auth, google-api-python-client, see ../requirements.txt)
are host-side-only and heavier than the rest of the stdlib-only host
scripts — CI's default job installs only llamaindex-service/requirements.txt
and must not start requiring these just to collect this test tree. Run with:

    python3 -m venv /tmp/gdrive_venv && /tmp/gdrive_venv/bin/pip install \\
        -r connectors/google_drive/requirements.txt pytest
    /tmp/gdrive_venv/bin/python -m pytest connectors/google_drive/tests -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
