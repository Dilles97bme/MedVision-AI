"""
==============================================================
MedVision-AI

Database Operations

Provides CRUD operations for prediction history.



==============================================================
"""

from app.database.connection import get_connection


def insert_prediction(
    image_name: str,
    prediction: str,
    confidence: float,
    normal_probability: float,
    pneumonia_probability: float,
    model_name: str,
    report_path: str,
    created_at: str,
) -> int:
    """
    Insert a prediction record into the database.

    Returns
    -------
    int
        ID of the inserted record.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (
            image_name,
            prediction,
            confidence,
            normal_probability,
            pneumonia_probability,
            model_name,
            report_path,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            image_name,
            prediction,
            confidence,
            normal_probability,
            pneumonia_probability,
            model_name,
            report_path,
            created_at,
        ),
    )

    connection.commit()

    prediction_id = cursor.lastrowid

    connection.close()

    return prediction_id


def get_prediction_history():
    """
    Retrieve all prediction records.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM predictions
        ORDER BY created_at DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return [dict(row) for row in rows]


def get_prediction_by_id(prediction_id: int):
    """
    Retrieve a prediction by its ID.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM predictions
        WHERE id = ?
        """,
        (prediction_id,),
    )

    row = cursor.fetchone()

    connection.close()

    return dict(row) if row else None


def delete_prediction(prediction_id: int):
    """
    Delete a prediction record.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM predictions
        WHERE id = ?
        """,
        (prediction_id,),
    )

    connection.commit()

    deleted = cursor.rowcount > 0

    connection.close()

    return deleted
