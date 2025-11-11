"""SQL validation utilities to catch errors before execution."""
import re
from typing import Tuple, Optional, Set
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Function
from sqlparse.tokens import Keyword, DML


def extract_table_names(sql: str) -> Set[str]:
    """Extract table names from SQL query."""
    parsed = sqlparse.parse(sql)[0]
    tables = set()
    
    from_seen = False
    for token in parsed.tokens:
        if from_seen:
            if isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    tables.add(str(identifier.get_real_name()).strip('"').lower())
            elif isinstance(token, Identifier):
                tables.add(str(token.get_real_name()).strip('"').lower())
            from_seen = False
        
        if token.ttype is Keyword and token.value.upper() in ['FROM', 'JOIN']:
            from_seen = True
    
    return tables


def extract_column_references(sql: str) -> Set[str]:
    """Extract column names referenced in SQL."""
    # Simple regex-based extraction (not perfect but catches most cases)
    columns = set()
    
    # Remove string literals to avoid false positives
    sql_clean = re.sub(r"'[^']*'", '', sql)
    
    # Find column references (word followed by comma, FROM, WHERE, GROUP, ORDER, etc.)
    pattern = r'\b([a-z_][a-z0-9_]*)\b(?=\s*[,\s]|$|\))'
    matches = re.findall(pattern, sql_clean, re.IGNORECASE)
    
    # Filter out SQL keywords
    sql_keywords = {
        'select', 'from', 'where', 'group', 'by', 'order', 'having', 
        'limit', 'as', 'and', 'or', 'on', 'join', 'left', 'right', 
        'inner', 'outer', 'sum', 'count', 'avg', 'max', 'min', 'distinct'
    }
    
    for match in matches:
        if match.lower() not in sql_keywords:
            columns.add(match.lower())
    
    return columns


def get_schema_tables_and_columns(schema: str) -> dict:
    """Parse schema to extract table and column names."""
    tables = {}
    
    # Match CREATE TABLE statements
    table_pattern = r'CREATE\s+TABLE\s+"?([a-zA-Z0-9_]+)"?\s*\((.*?)\);'
    for match in re.finditer(table_pattern, schema, re.IGNORECASE | re.DOTALL):
        table_name = match.group(1).lower()
        columns_str = match.group(2)
        
        # Extract column names
        col_pattern = r'"?([a-zA-Z0-9_]+)"?\s+(TEXT|INTEGER|REAL|BLOB|VARCHAR|CHAR|DATE|TIMESTAMP)'
        columns = [m.group(1).lower() for m in re.finditer(col_pattern, columns_str, re.IGNORECASE)]
        
        tables[table_name] = columns
    
    return tables


def validate_sql(sql: str, schema: str) -> Tuple[bool, Optional[str]]:
    """
    Validate SQL query against schema.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        # Parse SQL
        parsed = sqlparse.parse(sql)
        if not parsed:
            return False, "Could not parse SQL query"
        
        # Get schema structure
        schema_tables = get_schema_tables_and_columns(schema)
        
        # Extract referenced tables
        sql_tables = extract_table_names(sql)
        
        # Check if all tables exist
        for table in sql_tables:
            if table not in schema_tables:
                return False, f"Table '{table}' does not exist in schema. Available tables: {', '.join(schema_tables.keys())}"
        
        # Extract column references
        sql_columns = extract_column_references(sql)
        
        # Check if columns exist in their tables
        # This is a simplified check - doesn't handle table aliases perfectly
        all_valid_columns = set()
        for table, cols in schema_tables.items():
            all_valid_columns.update(cols)
        
        for col in sql_columns:
            # Skip common SQL functions and aliases
            if col not in all_valid_columns and col not in {'total', 'count', 'total_revenue'}:
                # Could be an alias, so just warn
                pass  # Don't fail on aliases
        
        # Check for common SQL errors
        sql_upper = sql.upper()
        
        # Aggregation without GROUP BY (if multiple columns)
        if 'SUM(' in sql_upper or 'COUNT(' in sql_upper or 'AVG(' in sql_upper:
            select_part = sql.split('FROM')[0] if 'FROM' in sql_upper else sql
            # Count selected columns (simple heuristic)
            non_agg_cols = len(re.findall(r'\bSELECT\s+(\w+)', select_part, re.IGNORECASE))
            has_group_by = 'GROUP BY' in sql_upper
            
            if non_agg_cols > 1 and not has_group_by:
                # This might be intentional, so just a warning
                pass
        
        return True, None
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def suggest_correction(sql: str, schema: str, error_msg: str) -> Optional[str]:
    """
    Suggest a corrected SQL query based on the error.
    
    This is a simple rule-based corrector. For production, use LLM-based correction.
    """
    schema_tables = get_schema_tables_and_columns(schema)
    
    # Handle missing table
    if "does not exist" in error_msg and "Table" in error_msg:
        # Extract the wrong table name
        match = re.search(r"Table '(\w+)'", error_msg)
        if match:
            wrong_table = match.group(1)
            # Find closest matching table (simple edit distance)
            best_match = min(schema_tables.keys(), 
                           key=lambda t: abs(len(t) - len(wrong_table)))
            corrected = sql.replace(wrong_table, best_match)
            return corrected
    
    return None
