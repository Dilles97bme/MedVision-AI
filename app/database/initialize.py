"""
==============================================================
MedVision-AI

Database Initialization

Creates required database tables.

==============================================================
"""

from app.database.connection import get_connection


def initialize_database() -> None:
    """
    Create database tables if they do not exist.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            image_name TEXT NOT NULL,

            prediction TEXT NOT NULL,

            confidence REAL NOT NULL,

            normal_probability REAL NOT NULL,

            pneumonia_probability REAL NOT NULL,

            model_name TEXT NOT NULL,

            report_path TEXT,

            created_at TEXT NOT NULL

        );
        """
    )

    connection.commit()

    connection.close()
