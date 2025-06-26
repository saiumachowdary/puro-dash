from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import sqlite3
from datetime import datetime, timedelta

router = APIRouter(prefix="/analytics", tags=["analytics"])

def get_db_connection():
    conn = sqlite3.connect('purogene.db')
    conn.row_factory = sqlite3.Row
    return conn

@router.get("/summary")
async def get_analytics_summary():
    """Get summary statistics of queries"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get total queries
        cursor.execute("SELECT COUNT(*) as count FROM query_log")
        total_queries = cursor.fetchone()['count']
        
        # Get successful queries
        cursor.execute("SELECT COUNT(*) as count FROM query_log WHERE status = 'success'")
        successful_queries = cursor.fetchone()['count']
        
        # Get average execution time
        cursor.execute("SELECT AVG(execution_time) as avg_time FROM query_log WHERE status = 'success'")
        avg_time = cursor.fetchone()['avg_time'] or 0
        
        # Get recent queries
        cursor.execute("""
            SELECT question, sql_query, status, execution_time, created_at 
            FROM query_log 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        recent_queries = [dict(row) for row in cursor.fetchall()]
        
        return {
            "total_queries": total_queries,
            "success_rate": (successful_queries / total_queries * 100) if total_queries > 0 else 0,
            "average_response_time": round(avg_time, 2),
            "recent_queries": recent_queries
        }
    finally:
        conn.close()

@router.get("/query-distribution")
async def get_query_distribution():
    """Get query distribution over time"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get query counts by hour for the last 24 hours
        cursor.execute("""
            SELECT 
                strftime('%H', created_at) as hour,
                COUNT(*) as count
            FROM query_log
            WHERE created_at >= datetime('now', '-1 day')
            GROUP BY hour
            ORDER BY hour
        """)
        
        distribution = [dict(row) for row in cursor.fetchall()]
        return {"distribution": distribution}
    finally:
        conn.close() 