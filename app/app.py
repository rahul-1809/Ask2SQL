import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

from .database_utils import create_sqlite_db_from_files, get_combined_schema, safe_execute_sql, get_table_summaries
from .sql_workflow_simplified import generate_sql_simplified, explain_sql_query

APP_ROOT = os.path.dirname(__file__)
DB_PATH = os.path.join(APP_ROOT, 'uploaded_data.db')
UPLOAD_DIR = os.path.join(APP_ROOT, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR

_session_state = {
    'schema': '',
    'history': []  # list of dicts {question, sql, rows, columns}
}


@app.route('/', methods=['GET'])
def index():
    # Get table summaries if database exists
    table_summaries = get_table_summaries(DB_PATH) if os.path.exists(DB_PATH) else []
    return render_template('index.html', schema=_session_state.get('schema', ''), table_summaries=table_summaries)


@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files')
    saved_paths = []
    filenames = []
    
    for f in files:
        if f.filename:
            filename = secure_filename(f.filename)
            path = os.path.join(UPLOAD_DIR, filename)
            f.save(path)
            saved_paths.append(path)
            filenames.append(filename)
    
    if not saved_paths:
        return 'No files uploaded', 400
    
    # Use append=True to keep existing tables
    tables, schema_text = create_sqlite_db_from_files(saved_paths, DB_PATH, append=True)
    _session_state['schema'] = schema_text
    
    # Get table summaries
    table_summaries = get_table_summaries(DB_PATH)
    
    # Create a nice message with details
    file_list = ', '.join([f"<strong>{fn}</strong>" for fn in filenames])
    table_list = ', '.join([f"<strong>{t}</strong>" for t in tables])
    
    # Get all existing tables
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
    all_tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    all_table_list = ', '.join([f"<strong>{t}</strong>" for t in all_tables])
    message = f'''
    <div style="margin-bottom:8px">✅ Successfully uploaded {len(saved_paths)} file(s): {file_list}</div>
    <div style="margin-bottom:8px">📊 Added/Updated {len(tables)} table(s): {table_list}</div>
    <div>💾 Total tables in database: {len(all_tables)} ({all_table_list})</div>
    '''
    
    return render_template('index.html', schema=schema_text, message=message, table_summaries=table_summaries)


@app.route('/query', methods=['POST'])
def query():
    question = request.form.get('question', '').strip()
    if not question:
        return 'Question required', 400
    schema_text = _session_state.get('schema', '')
    if not schema_text:
        return 'No schema loaded. Upload files first.', 400

    # Use simplified workflow (single unified approach)
    print("🔧 Using simplified LangGraph workflow")
    try:
        sql, agent_logs, is_valid = generate_sql_simplified(
            question=question,
            schema=schema_text,
            max_retries=3,
            verbose=False
        )
        warning = None if is_valid else "⚠️ SQL generated with validation warnings"
        suggested_sql = None
        
        # Generate explanation
        explanation = explain_sql_query(sql, question, schema_text)
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        # Emergency fallback
        sql = f"SELECT 'Error: {str(e)}' as error"
        is_valid = False
        warning = f"❌ Workflow error: {e}"
        suggested_sql = None
        explanation = None

    # Execute
    try:
        columns, rows = safe_execute_sql(DB_PATH, sql)
        row_dicts = [dict(zip(columns, r)) for r in rows]
        summary = f"Returned {len(rows)} rows."
        viz = None
        _session_state['history'].append({'question': question, 'sql': sql, 'rows': row_dicts, 'columns': columns, 'explanation': explanation})
        return render_template('results.html', question=question, sql=sql, columns=columns, rows=rows, summary=summary, viz=viz, warning=warning, suggested_sql=suggested_sql, explanation=explanation)
    except Exception as e:
        fix = f"Error: {str(e)}"
        return render_template('error.html', question=question, sql=sql, error=str(e), fix=fix)


@app.route('/history', methods=['GET'])
def history():
    return render_template('history.html', history=_session_state['history'])


@app.route('/clear', methods=['POST'])
def clear_database():
    """Clear all tables from the database"""
    import sqlite3
    
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    
    _session_state['schema'] = ''
    _session_state['history'] = []
    
    message = '🗑️ Database cleared! All tables have been removed. Upload new files to get started.'
    return render_template('index.html', schema='', message=message, table_summaries=[])


if __name__ == '__main__':
    app.run(debug=True)
