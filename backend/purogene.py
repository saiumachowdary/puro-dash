import os
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import pandas as pd
import mysql.connector
import ollama
import re
import time
import sqlalchemy
import warnings
import logging
from fastapi.middleware.cors import CORSMiddleware
from .analytics import router as analytics_router
import sqlite3
import json
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas.io.sql")

# Create FastAPI app instance
app = FastAPI(
    title="NL to SQL API",
    description="Convert natural language queries to SQL and execute them",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the analytics router
app.include_router(analytics_router)

# MySQL Connection Details
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'purogene',
    'port': 3306
}

# Initialize SQLite database
def init_db():
    """Initialize MySQL database with required tables"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Create permissions table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS permissions (
                permission_id INT NOT NULL AUTO_INCREMENT,
                title_label VARCHAR(100) NOT NULL,
                enu_name VARCHAR(150) NOT NULL,
                type INT NOT NULL,
                label VARCHAR(150),
                createdAt DATETIME NOT NULL,
                updatedAt DATETIME NOT NULL,
                PRIMARY KEY (permission_id)
            )
        ''')
        
        # Create query_log table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_log (
                id INT NOT NULL AUTO_INCREMENT,
                question TEXT NOT NULL,
                sql_query TEXT NOT NULL,
                status VARCHAR(50) NOT NULL,
                execution_time FLOAT NOT NULL,
                error_message TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id)
            )
        ''')
        
        # Insert sample data if permissions table is empty
        cursor.execute("SELECT COUNT(*) FROM permissions")
        if cursor.fetchone()[0] == 0:
            cursor.execute('''
                INSERT INTO permissions 
                (title_label, enu_name, type, label, createdAt, updatedAt)
                VALUES 
                ('Admin Access', 'ADMIN_ACCESS', 1, 'Administrator Access', NOW(), NOW()),
                ('User Access', 'USER_ACCESS', 2, 'Regular User Access', NOW(), NOW())
            ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        logging.info("Database initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing database: {str(e)}")
        raise

# Initialize database on startup
init_db()

# Hardcoded schema for OSW tables
SCHEMA = """
Table:permissions
  permission_id int NOT NULL,
  title_label varchar(100)  NOT NULL,
  menu_name varchar(150)  NOT NULL,
  type int NOT NULL,
  label varchar(150),
  createdAt datetime NOT NULL,
  updatedAt datetime NOT NULL
TABLE transaction_numbers 
   id int NOT NULL,
   module text NOT NULL,
   prefix varchar(25) NOT NULL,
   year varchar(10) NOT NULL,
   starting_number varchar(10) NOT NULL,
   restart_numbering enum('Yearly','None') NOT NULL,
   createdAt datetime NOT NULL,
   updatedAt datetime NOT NULL
TABLE  support_queries
   id  int NOT NULL,
   emp_id int DEFAULT NULL,
   title text,
   description text,
   image  text,
   priority enum('Low','Normal','High','Urgent') NOT NULL DEFAULT 'Normal',
   status enum('New','Open','Done','Closed') NOT NULL DEFAULT 'New',
   createdAt datetime NOT NULL,
   updatedAt datetime NOT NULL
TABLE users
   user_id int NOT NULL,
   email varchar(100)  NOT NULL,
   phone varchar(15)  DEFAULT NULL,
   is_active tinyint(1) NOT NULL DEFAULT '1',
   username varchar(225)  DEFAULT NULL,
   password varchar(225)  NOT NULL,
   role_id  int NOT NULL,
   is_deleted tinyint(1) NOT NULL DEFAULT '0',
   createdAt datetime NOT NULL,
   updatedAt datetime NOT NULL


"""

# Common aggregate patterns to check in validation
AGGREGATE_FUNCTIONS = ["COUNT", "SUM", "AVG", "MIN", "MAX", "GROUP_CONCAT", "STD", "STDDEV", "VARIANCE"]

class QueryRequest(BaseModel):
    question: str
    execute_query: bool = True  # Flag to determine if SQL should be executed
    request_id: Optional[str] = None  # Optional request ID for tracking

class QueryResponse(BaseModel):
    sql_query: str
    generation_time: float
    data: Optional[List[Dict[str, Any]]] = None
    execution_time: Optional[float] = None
    row_count: Optional[int] = None
    error: Optional[str] = None
    request_id: Optional[str] = None  # Return the request ID for tracking

def is_aggregate_query(question: str) -> bool:
    """Check if the natural language question likely involves aggregation"""
    aggregate_keywords = [
        "average", "avg", "mean", "median", "sum", "total", "count", "how many", 
        "maximum", "minimum", "max", "min", "group by", "grouped by", "distribution",
        "percentage", "ratio", "proportion", "statistics", "breakdown", "summarize",
        "standard deviation", "variance", "aggregated", "per", "each"
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in aggregate_keywords)

def generate_sql(question: str, schema: str) -> tuple:
    """Generate SQL query from natural language using Ollama"""
    try:
        # Normalize schema to lowercase for better match with natural language
        normalized_schema = '\n'.join(line.lower() for line in schema.splitlines())
        
        # Determine if this is likely an aggregate query
        is_aggregate = is_aggregate_query(question)
        
        # Add specific guidance for aggregate functions if the query seems to need it
        aggregate_guidance = ""
        if is_aggregate:
            aggregate_guidance = """
6. For aggregation queries:
   - Use appropriate aggregate functions (COUNT, SUM, AVG, MIN, MAX) based on the question.
   - Include appropriate GROUP BY clauses when grouping data.
   - Consider using HAVING clauses for conditions on aggregate results.
   - Always include meaningful column names or aliases for aggregate functions.
   - For comparisons or rankings, consider using subqueries or window functions if needed.
   - IMPORTANT: MySQL requires that columns appearing in the SELECT list that are not aggregated must appear in the GROUP BY clause."""

        prompt = f"""### MySQL SQL Query Generator
Given the database schema below, generate a syntactically correct MySQL query to answer the question.
Only return the SQL query, no explanations, comments, or markdown formatting.

Database Schema:
{normalized_schema}

User Question: {question}

Important Notes:
1. Use column and table names exactly as shown in the schema.
2. If the question asks for something like understanding or other values that require identifying members, include their unique identifier (e.g., member_id).
3. If the question asks for a member's details (e.g., personal spirit), make sure to include the necessary identifying columns (e.g., member_id, first_name, last_name) for clarity and relevance.
4. Avoid using JOINs unless absolutely necessary, such as when the question explicitly requires data from multiple tables.
5. Only return raw SQL with no extra formatting or markdown.{aggregate_guidance}
"""

        logging.info(f"Generating SQL for question: {question}")
        start_time = time.time()
        response = ollama.chat(
            model="deepseek-coder:6.7b-instruct",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7}
        )
        end_time = time.time()

        content = response['message']['content']

        # Extract and clean SQL
        if "```sql" in content:
            content = content.split("```sql")[-1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()

        content = re.sub(r'--.*$', '', content, flags=re.MULTILINE).strip()
        
        # Validate that aggregate queries have appropriate GROUP BY clauses
        if is_aggregate and any(f"{func}(" in content.upper() for func in AGGREGATE_FUNCTIONS):
            if "GROUP BY" not in content.upper() and re.search(r'SELECT.*,.*FROM', content, re.IGNORECASE | re.DOTALL):
                # Likely missing GROUP BY with multiple columns selected
                logging.warning("Generated aggregate query might be missing required GROUP BY clause")
        
        generation_time = end_time - start_time
        
        logging.info(f"Generated SQL: {content}")
        logging.info(f"Generation time: {generation_time:.2f}s")

        return content, generation_time
    except Exception as e:
        logging.error(f"Error generating SQL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating SQL: {str(e)}")

def execute_sql(query: str) -> tuple:
    """Execute SQL query and return results as a pandas DataFrame"""
    try:
        # Create a new SQLAlchemy engine for each query
        engine = sqlalchemy.create_engine(
            f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}",
            pool_recycle=3600, 
            pool_pre_ping=True
        )
        
        logging.info(f"Executing SQL query: {query}")
        # Execute the query with a fresh connection
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
        
        logging.info(f"Query executed successfully. Row count: {len(df)}")
        return df, None
    except Exception as e:
        logging.error(f"Error with SQLAlchemy execution: {str(e)}")
        try:
            # Fallback to direct connector if SQLAlchemy fails
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query)
            results = cursor.fetchall()
            df = pd.DataFrame(results)
            cursor.close()
            conn.close()  # Explicitly close connection
            return df, None
        except Exception as e2:
            logging.error(f"Error with direct connector: {str(e2)}")
            return None, str(e2)

def fix_sql_query(query: str, error_msg: str) -> str:
    """Fix a SQL query that produced an error, with special handling for aggregate functions"""
    try:
        logging.info(f"Attempting to fix SQL query: {query}")
        logging.info(f"Error message: {error_msg}")
        
        # Detect common aggregate function errors
        aggregate_hint = ""
        
        # Check for common aggregate errors
        if "not in GROUP BY" in error_msg:
            aggregate_hint = """
This appears to be an error with aggregate functions and GROUP BY clauses.
Make sure:
1. All non-aggregated columns in the SELECT list are included in the GROUP BY clause
2. Or wrap non-grouped columns in aggregate functions like MAX() or MIN()
3. Consider using ANY_VALUE() function for columns that don't affect aggregation results
"""
        elif "invalid use of group function" in error_msg.lower():
            aggregate_hint = """
This appears to be an error with nested aggregate functions or using aggregates in WHERE.
Make sure:
1. Aggregate functions aren't nested (e.g., AVG(SUM(column)))
2. Aggregate functions aren't used in WHERE clauses (use HAVING instead)
3. Consider using subqueries for complex aggregation requirements
"""
        
        fix_prompt = f"""The following SQL query has an error:

{query}

Error message: {error_msg}

{aggregate_hint}

Please fix the SQL query to work with the given schema:
{SCHEMA}

Important guidelines:
1. Only use JOINs if absolutely necessary to answer the question.
2. Use proper table names and column names as shown in the schema.
3. For aggregate queries, ensure all non-aggregated columns in SELECT are included in GROUP BY.
4. Use HAVING instead of WHERE for conditions on aggregated values.
5. Only return the corrected SQL query, nothing else."""

        response = ollama.chat(
            model="deepseek-coder:6.7b-instruct",
            messages=[{"role": "user", "content": fix_prompt}],
            options={"temperature": 0.7}
        )
        
        fixed_sql = response['message']['content']
        if "```sql" in fixed_sql:
            fixed_sql = fixed_sql.split("```sql")[-1].split("```")[0].strip()
        elif "```" in fixed_sql:
            fixed_sql = fixed_sql.split("```")[1].strip()
        
        logging.info(f"Fixed SQL query: {fixed_sql}")
        return fixed_sql
    except Exception as e:
        logging.error(f"Error fixing SQL query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fixing SQL query: {str(e)}")

def postprocess_aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Apply post-processing to aggregate results for better presentation"""
    # Convert numeric strings to actual numeric types where appropriate
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Check if column contains numeric data as strings
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if not numeric_col.isna().all():
                    df[col] = numeric_col
            except:
                pass
    
    # Sort data if it appears to be a frequency distribution or ranking
    if len(df.columns) == 2 and df.shape[0] > 1:
        # If two columns and one is likely numeric, sort by it
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df = df.sort_values(by=col, ascending=False)
                break
    
    return df

def get_db_connection():
    conn = sqlite3.connect('purogene.db')
    conn.row_factory = sqlite3.Row
    return conn

def log_query(question: str, sql_query: str, status: str, execution_time: float, error_message: str = None):
    """Log query execution details to MySQL database"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO query_log 
            (question, sql_query, status, execution_time, error_message)
            VALUES (%s, %s, %s, %s, %s)
        ''', (question, sql_query, status, execution_time, error_message))
        
        conn.commit()
        cursor.close()
        conn.close()
        logging.info(f"Query logged successfully. Status: {status}")
    except Exception as e:
        logging.error(f"Error logging query: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def nl_to_sql_query(request: QueryRequest):
    """Convert natural language to SQL query and optionally execute it"""
    request_id = request.request_id or str(uuid.uuid4())
    logging.info(f"Received request ID: {request_id}")
    logging.info(f"Received question: {request.question}")
    
    start_time = time.time()
    try:
        # Generate SQL query
        logging.info(f"Generating SQL for question: {request.question}")
        sql_query, generation_time = generate_sql(request.question, SCHEMA)
        logging.info(f"Generated SQL: {sql_query}")
        logging.info(f"Generation time: {generation_time:.2f}s")
        
        # Execute query if requested
        if request.execute_query:
            logging.info(f"Executing SQL query: {sql_query}")
            df, error = execute_sql(sql_query)
            execution_time = time.time() - start_time
            
            if error:
                # Try to fix the query if it has errors
                try:
                    fixed_sql = fix_sql_query(sql_query, error)
                    start_time = time.time()
                    df, error = execute_sql(fixed_sql)
                    execution_time = time.time() - start_time
                    
                    if not error:
                        # Fixed query succeeded
                        sql_query = fixed_sql
                        
                        # Apply post-processing for aggregate queries
                        if is_aggregate_query(request.question):
                            df = postprocess_aggregate_results(df)
                            
                        response = QueryResponse(
                            request_id=request_id,
                            sql_query=sql_query,
                            generation_time=generation_time,
                            execution_time=execution_time,
                            data=df.to_dict(orient='records') if not df.empty else [],
                            row_count=len(df)
                        )
                    else:
                        # Still had an error after fixing
                        response = QueryResponse(
                            request_id=request_id,
                            error=f"Original error: {error}. Unable to fix query automatically."
                        )
                except Exception as e:
                    response = QueryResponse(
                        request_id=request_id,
                        error=f"Original error: {error}. Error while trying to fix: {str(e)}"
                    )
            else:
                # Query executed successfully
                # Apply post-processing for aggregate queries
                if is_aggregate_query(request.question):
                    df = postprocess_aggregate_results(df)
                    
                response = QueryResponse(
                    request_id=request_id,
                    sql_query=sql_query,
                    generation_time=generation_time,
                    execution_time=execution_time,
                    data=df.to_dict(orient='records') if not df.empty else [],
                    row_count=len(df)
                )
            
            # Log successful query
            log_query(
                question=request.question,
                sql_query=sql_query,
                status="success",
                execution_time=execution_time
            )
        else:
            # Log query without execution
            log_query(
                question=request.question,
                sql_query=sql_query,
                status="executed",
                execution_time=generation_time
            )
        
        return response
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logging.error(error_msg)
        log_query(
            request.question,
            sql_query if 'sql_query' in locals() else "Failed to generate SQL",
            "error",
            time.time() - start_time,
            error_msg
        )
        return QueryResponse(
            request_id=request_id,
            error=error_msg
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 