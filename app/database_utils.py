import os
import re
import sqlite3
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd


def _normalize_table_name(name: str) -> str:
    # Replace spaces and special chars with underscores, lower-case for consistency
    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())
    safe = re.sub(r"_+", "_", safe)
    return safe.strip("_").lower() or "table"


def create_sqlite_db_from_files(file_paths: List[str], db_path: str, append: bool = False) -> Tuple[List[str], str]:
    """
    Create/update a SQLite DB from a list of Excel/CSV files. Each sheet (Excel) or file (CSV) becomes a table.
    
    Args:
        file_paths: List of file paths to load
        db_path: Path to SQLite database
        append: If True, add tables to existing DB. If False, replace entire DB.

    Returns (table_names, combined_schema_text)
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Only remove existing DB if append=False
    if not append and os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    try:
        table_names: List[str] = []

        for file_path in file_paths:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in {".xlsx", ".xls"}:
                # Read all sheets
                sheets = pd.read_excel(file_path, sheet_name=None)
                for sheet_name, df in sheets.items():
                    table = _normalize_table_name(sheet_name)
                    df.columns = [_normalize_table_name(str(c)) for c in df.columns]
                    df.to_sql(table, conn, if_exists="replace", index=False)
                    table_names.append(table)
            elif ext in {".csv"}:
                df = pd.read_csv(file_path)
                df.columns = [_normalize_table_name(str(c)) for c in df.columns]
                table = _normalize_table_name(os.path.splitext(os.path.basename(file_path))[0])
                df.to_sql(table, conn, if_exists="replace", index=False)
                table_names.append(table)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")

        schema_text = get_combined_schema(db_path)
        return sorted(set(table_names)), schema_text
    finally:
        conn.close()


def get_combined_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
        rows = cur.fetchall()
        schema_parts = []
        for name, sql in rows:
            if sql:
                schema_parts.append(sql.strip().rstrip(';') + ';')
        return "\n\n".join(schema_parts)
    finally:
        conn.close()


def get_table_summaries(db_path: str) -> List[Dict[str, Any]]:
    """
    Get summary information for all tables in the database.
    Returns list of dicts with table metadata and sample data.
    """
    if not os.path.exists(db_path):
        return []
    
    conn = sqlite3.connect(db_path)
    summaries = []
    
    try:
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table})")
            columns_info = cursor.fetchall()
            columns = [{'name': col[1], 'type': col[2]} for col in columns_info]
            
            # Get sample data (first 3 rows)
            cursor.execute(f"SELECT * FROM {table} LIMIT 3")
            sample_rows = cursor.fetchall()
            column_names = [col['name'] for col in columns]
            
            # Get basic statistics for numeric columns
            stats = {}
            for col in columns:
                col_name = col['name']
                col_type = col['type'].upper()
                
                # Try to get stats for numeric-looking columns
                if any(t in col_type for t in ['INT', 'REAL', 'FLOAT', 'NUMERIC', 'DECIMAL']):
                    try:
                        cursor.execute(f"SELECT MIN({col_name}), MAX({col_name}), AVG({col_name}) FROM {table}")
                        min_val, max_val, avg_val = cursor.fetchone()
                        stats[col_name] = {
                            'min': min_val,
                            'max': max_val,
                            'avg': round(avg_val, 2) if avg_val else None
                        }
                    except:
                        pass  # Skip if column isn't actually numeric
            
            summaries.append({
                'table_name': table,
                'row_count': row_count,
                'column_count': len(columns),
                'columns': columns,
                'sample_rows': sample_rows,
                'column_names': column_names,
                'stats': stats
            })
    
    finally:
        conn.close()
    
    return summaries


def safe_execute_sql(db_path: str, sql: str, params: Optional[Tuple[Any, ...]] = None, max_rows: int = 1000) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    """
    Execute a read-only SELECT query safely. Returns (columns, rows).
    - Only allows a single SELECT statement.
    - Limits result rows to max_rows.
    """
    # Basic safety checks
    if not isinstance(sql, str):
        raise ValueError("SQL must be a string")

    stripped = sql.strip().strip(';')
    # Disallow multiple statements
    if ';' in stripped:
        raise ValueError("Only a single SELECT statement is allowed")

    # Must start with SELECT (case-insensitive) and not contain UPDATE/DELETE/INSERT/PRAGMA/etc.
    if not re.match(r"^\s*select\b", stripped, flags=re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed")

    forbidden = re.compile(r"\b(update|delete|insert|drop|alter|truncate|attach|detach|reindex|vacuum|pragma)\b", re.IGNORECASE)
    if forbidden.search(stripped):
        raise ValueError("Query contains forbidden statements")

    # Add LIMIT if missing
    if re.search(r"\blimit\b", stripped, flags=re.IGNORECASE) is None:
        stripped += f" LIMIT {int(max_rows)}"

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        if params:
            cur.execute(stripped, params)
        else:
            cur.execute(stripped)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
        return columns, rows
    finally:
        conn.close()
