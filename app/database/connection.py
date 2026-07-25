"""
==============================================================
MedVision-AI

Database Connection

Provides SQLite database connection.


"""

from pathlib import Path
import sqlite3

# ----------------------------------------------------------
# Database Path
# ----------------------------------------------------------

DATABASE_DIR = Path("database")
DATABASE_DIR.mkdir(exist_ok=True)

DATABASE_PATH = DATABASE_DIR / "prediction_history.db"


# ----------------------------------------------------------
# Connection
# ----------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """
    Return a SQLite database connection.

    Returns
    -------
    sqlite3.Connection
        Active database connection.
    """

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    return connection
