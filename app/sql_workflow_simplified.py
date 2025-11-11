"""
SIMPLIFIED LANGGRAPH WORKFLOW
Single unified approach using both fine-tuned model and Groq intelligently

Architecture:
- Classifier determines if query is SIMPLE or COMPLEX
- SIMPLE → Fine-tuned CodeT5 (free, unlimited)
- COMPLEX → Groq API (high accuracy)
- Both go through same validation/correction loop
"""

import os
import re
from typing import TypedDict, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# Handle imports
try:
    from .sql_validator import validate_sql, get_schema_tables_and_columns
    from .query_classifier import classify_query, QueryComplexity
except ImportError:
    from sql_validator import validate_sql, get_schema_tables_and_columns
    from query_classifier import classify_query, QueryComplexity

# Try to import transformers and PEFT for fine-tuned model
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel, PeftConfig
    TRANSFORMERS_AVAILABLE = True
    PEFT_AVAILABLE = True
except ImportError:
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        TRANSFORMERS_AVAILABLE = True
        PEFT_AVAILABLE = False
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        PEFT_AVAILABLE = False


# ============================================================================
# State Definition
# ============================================================================

class SimplifiedSQLState(TypedDict):
    """Unified state for streamlined workflow"""
    
    # Inputs
    question: str
    schema: str
    
    # Classification
    complexity: str  # "SIMPLE" or "COMPLEX"
    
    # Generated SQL
    sql: Optional[str]
    
    # Validation
    is_valid: bool
    validation_errors: list[str]
    
    # Control
    retry_count: int
    max_retries: int
    
    # Observability
    logs: list[str]


# ============================================================================
# Fine-tuned Model Loader (cached)
# ============================================================================

_fine_tuned_model = None
_fine_tuned_tokenizer = None

def load_fine_tuned_model():
    """Load fine-tuned CodeT5 model with LoRA adapter (cached)"""
    global _fine_tuned_model, _fine_tuned_tokenizer
    
    if _fine_tuned_model is not None:
        return _fine_tuned_tokenizer, _fine_tuned_model
    
    if not TRANSFORMERS_AVAILABLE:
        print("⚠️  Transformers not available, cannot load fine-tuned model")
        return None, None
    
    # Path to LoRA adapter
    adapter_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "models", 
        "fine_tuned_text2sql_codet5"
    )
    
    try:
        # Check if this is a LoRA adapter (has adapter_config.json)
        if os.path.isdir(adapter_dir) and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            if not PEFT_AVAILABLE:
                print("⚠️  PEFT library not available. LoRA adapter found but cannot be loaded.")
                print("   Install with: pip install peft")
                return None, None
            
            print(f"📦 Loading LoRA adapter from: {adapter_dir}")
            
            # Load base model first
            base_model_name = "Salesforce/codet5-small"
            print(f"   Loading base model: {base_model_name}")
            _fine_tuned_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
            
            # Apply LoRA adapter
            print(f"   Applying LoRA adapter...")
            _fine_tuned_model = PeftModel.from_pretrained(base_model, adapter_dir)
            print(f"   ✓ Fine-tuned model loaded successfully!")
            
            return _fine_tuned_tokenizer, _fine_tuned_model
            
        # Check if this is a full model (has config.json)
        elif os.path.isdir(adapter_dir) and os.path.exists(os.path.join(adapter_dir, "config.json")):
            print(f"📦 Loading full model from: {adapter_dir}")
            _fine_tuned_tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
            _fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained(adapter_dir)
            print(f"   ✓ Full model loaded successfully!")
            return _fine_tuned_tokenizer, _fine_tuned_model
            
        else:
            # Fall back to base model
            print(f"⚠️  No fine-tuned model found at {adapter_dir}")
            print(f"   Falling back to base model: Salesforce/codet5-small")
            _fine_tuned_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
            _fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")
            return _fine_tuned_tokenizer, _fine_tuned_model
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None


# ============================================================================
# Agent 1: Classifier (Fast, Rule-based)
# ============================================================================

def classifier_agent(state: SimplifiedSQLState) -> SimplifiedSQLState:
    """Classify query as SIMPLE or COMPLEX"""
    
    complexity, details = classify_query(state["question"], state["schema"])
    
    state["complexity"] = complexity
    state["logs"].append(f"[Classifier] Query is {complexity} (score: {details['score']})")
    
    # Better logging with explanation
    if complexity == "SIMPLE":
        print(f"🔍 [CLASSIFIER] Query classified as SIMPLE")
        print(f"   Complexity score: {details['score']} (threshold: {details['threshold']})")
        print(f"   → Will use fine-tuned CodeT5 model (free, unlimited)")
    else:
        print(f"🔍 [CLASSIFIER] Query classified as COMPLEX")
        print(f"   Complexity score: {details['score']} (threshold: {details['threshold']})")
        print(f"   Indicators: {', '.join(details['indicators'][:3])}")
        print(f"   → Will use Groq API (high accuracy)")
    
    return state


# ============================================================================
# Agent 2: SQL Generator (Hybrid: Fine-tuned + Groq)
# ============================================================================

def generator_agent(state: SimplifiedSQLState) -> SimplifiedSQLState:
    """
    Generate SQL using appropriate method:
    - SIMPLE queries → Fine-tuned CodeT5 (free, unlimited)
    - COMPLEX queries → Groq API (high accuracy)
    """
    
    # Skip if already generated (retry_count > 0)
    if state.get("retry_count", 0) > 0:
        return state
    
    question = state["question"]
    schema = state["schema"]
    complexity = state.get("complexity", QueryComplexity.SIMPLE)
    
    sql = None
    
    # Route based on complexity
    if complexity == QueryComplexity.SIMPLE:
        # Try fine-tuned model first (free, unlimited)
        state["logs"].append("[Generator] Using fine-tuned CodeT5 model (SIMPLE query)")
        print("🤖 [SQL GENERATION] Using fine-tuned CodeT5 model for SIMPLE query")
        
        tokenizer, model = load_fine_tuned_model()
        
        if tokenizer and model:
            try:
                prompt = f"Schema: {schema}\nQuestion: {question}\nSQL:"
                inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                sql_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Extract SQL (everything after last 'SQL:')
                if 'SQL:' in sql_text:
                    sql = sql_text.split('SQL:')[-1].strip()
                else:
                    sql = sql_text.strip()
                
                # Clean up
                sql = sql.replace('\n', ' ').strip().rstrip(';')
                
                # Validate it looks like SQL
                if sql and sql.upper().startswith('SELECT'):
                    state["logs"].append(f"[Generator] Fine-tuned model: {sql[:80]}...")
                    print(f"   ✓ Generated SQL: {sql[:100]}...")
                else:
                    sql = None  # Fall through to Groq
                    print("   ✗ Fine-tuned model output invalid, falling back to Groq")
            except Exception as e:
                state["logs"].append(f"[Generator] Fine-tuned model failed: {e}")
                print(f"   ✗ Fine-tuned model error: {e}, falling back to Groq")
                sql = None
    
    # Fall back to Groq for COMPLEX queries or if fine-tuned failed
    if sql is None:
        state["logs"].append("[Generator] Using Groq API (COMPLEX query or fallback)")
        print("🚀 [SQL GENERATION] Using Groq API (COMPLEX query or fallback)")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            state["logs"].append("[Generator] ERROR: GROQ_API_KEY not set")
            state["sql"] = "SELECT 1"  # Emergency fallback
            return state
        
        try:
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.0,
                api_key=api_key,
                max_tokens=500
            )
            
            system_prompt = """You are an expert SQL generator. Generate ONLY valid SQLite queries.

CRITICAL RULES:
1. With aggregations (SUM, AVG, COUNT, MIN, MAX), MUST include GROUP BY for non-aggregated columns
2. For "average X for each Y": SELECT Y, AVG(X) ... GROUP BY Y
3. Use proper table/column names from schema
4. No markdown - just raw SQL
5. End with semicolon"""
            
            user_prompt = f"""Database Schema:
{schema}

Question: {question}

Generate the SQL query:"""
            
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            sql = response.content.strip()
            
            # Clean formatting
            if '```sql' in sql:
                sql = sql.split('```sql')[1].split('```')[0].strip()
            elif '```' in sql:
                sql = sql.split('```')[1].split('```')[0].strip()
            
            sql = sql.strip().rstrip(';')
            
            state["logs"].append(f"[Generator] Groq API: {sql[:80]}...")
            print(f"   ✓ Generated SQL: {sql[:100]}...")
        
        except Exception as e:
            state["logs"].append(f"[Generator] Groq failed: {e}")
            print(f"   ✗ Groq API error: {e}")
            sql = "SELECT 1"  # Emergency fallback
    
    state["sql"] = sql
    return state


# ============================================================================
# Agent 3: Validator (Same as before)
# ============================================================================

def validator_agent(state: SimplifiedSQLState) -> SimplifiedSQLState:
    """Validate SQL with comprehensive checks"""
    
    sql = state["sql"]
    schema = state["schema"]
    
    errors = []
    
    # Basic validation
    is_valid, error_msg = validate_sql(sql, schema)
    if not is_valid:
        errors.append(f"Schema validation: {error_msg}")
    
    # Check GROUP BY with aggregations
    sql_upper = sql.upper()
    has_aggregation = any(agg in sql_upper for agg in ['SUM(', 'AVG(', 'COUNT(', 'MIN(', 'MAX('])
    has_group_by = 'GROUP BY' in sql_upper
    
    if has_aggregation and not has_group_by:
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            if ',' in select_clause:
                parts = [p.strip() for p in select_clause.split(',')]
                non_agg_columns = []
                for part in parts:
                    if not any(f in part.upper() for f in ['SUM(', 'AVG(', 'COUNT(', 'MIN(', 'MAX(']):
                        col = part.split(' AS ')[0].strip() if ' AS ' in part.upper() else part.strip()
                        if col and col != '*' and not col.isdigit():
                            non_agg_columns.append(col)
                
                if non_agg_columns:
                    errors.append(
                        f"Missing GROUP BY: Query uses aggregations but SELECT includes "
                        f"non-aggregated columns: {', '.join(non_agg_columns)}. "
                        f"Add 'GROUP BY {', '.join(non_agg_columns)}'."
                    )
    
    state["is_valid"] = len(errors) == 0
    state["validation_errors"] = errors
    
    if errors:
        state["logs"].append(f"[Validator] Found {len(errors)} error(s)")
        for err in errors[:2]:  # Show first 2
            state["logs"].append(f"  - {err}")
    else:
        state["logs"].append("[Validator] SQL is valid ✓")
    
    return state


# ============================================================================
# Agent 4: Corrector (Groq only, for both SIMPLE and COMPLEX)
# ============================================================================

def corrector_agent(state: SimplifiedSQLState) -> SimplifiedSQLState:
    """Self-correct SQL using Groq (works for both SIMPLE and COMPLEX)"""
    
    if not state.get("validation_errors"):
        return state
    
    print(f"🔧 [CORRECTOR] Attempting to fix SQL using Groq API (attempt {state.get('retry_count', 0) + 1})")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        state["logs"].append("[Corrector] Cannot correct without GROQ_API_KEY")
        print("   ✗ Cannot correct: GROQ_API_KEY not set")
        return state
    
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=api_key,
            max_tokens=500
        )
        
        errors_text = "\n".join(f"- {err}" for err in state["validation_errors"][:3])
        
        system_prompt = """You are a SQL debugging expert. Fix the broken SQL query.

Focus on:
- Missing GROUP BY with aggregations
- Wrong table/column names
- Syntax errors

Return ONLY the corrected SQL, no explanations."""
        
        user_prompt = f"""Question: {state['question']}

Schema:
{state['schema']}

Broken SQL:
{state['sql']}

Errors:
{errors_text}

Generate the CORRECTED SQL:"""
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        corrected_sql = response.content.strip()
        
        # Clean formatting
        if '```sql' in corrected_sql:
            corrected_sql = corrected_sql.split('```sql')[1].split('```')[0].strip()
        elif '```' in corrected_sql:
            corrected_sql = corrected_sql.split('```')[1].split('```')[0].strip()
        
        corrected_sql = corrected_sql.strip().rstrip(';')
        
        state["sql"] = corrected_sql
        state["retry_count"] += 1
        state["logs"].append(f"[Corrector] Attempt {state['retry_count']}: Fixed SQL")
        print(f"   ✓ Corrected SQL: {corrected_sql[:100]}...")
    
    except Exception as e:
        state["logs"].append(f"[Corrector] Failed: {e}")
        print(f"   ✗ Correction failed: {e}")
    
    return state


# ============================================================================
# Control Flow
# ============================================================================

def should_retry(state: SimplifiedSQLState) -> Literal["retry", "end"]:
    """Decide retry or end"""
    if state["is_valid"]:
        return "end"
    if state.get("retry_count", 0) >= state.get("max_retries", 3):
        return "end"
    return "retry"


# ============================================================================
# Workflow Construction
# ============================================================================

def create_simplified_workflow() -> StateGraph:
    """
    Build streamlined workflow:
    
    START → Classifier → Generator → Validator → [Valid?]
                                                   ↓ No
                                                 Corrector → Validator
                                                   ↓ Yes
                                                  END
    """
    workflow = StateGraph(SimplifiedSQLState)
    
    # Add agents
    workflow.add_node("classifier", classifier_agent)
    workflow.add_node("generator", generator_agent)
    workflow.add_node("validator", validator_agent)
    workflow.add_node("corrector", corrector_agent)
    
    # Flow
    workflow.set_entry_point("classifier")
    workflow.add_edge("classifier", "generator")
    workflow.add_edge("generator", "validator")
    
    workflow.add_conditional_edges(
        "validator",
        should_retry,
        {
            "retry": "corrector",
            "end": END
        }
    )
    
    workflow.add_edge("corrector", "validator")
    
    return workflow.compile()


# ============================================================================
# Main Interface
# ============================================================================

def generate_sql_simplified(
    question: str,
    schema: str,
    max_retries: int = 3,
    verbose: bool = False
) -> tuple[str, list[str], bool]:
    """
    Generate SQL using simplified unified workflow.
    
    Returns: (sql, logs, is_valid)
    """
    workflow = create_simplified_workflow()
    
    initial_state = SimplifiedSQLState(
        question=question,
        schema=schema,
        complexity=QueryComplexity.SIMPLE,
        sql=None,
        is_valid=False,
        validation_errors=[],
        retry_count=0,
        max_retries=max_retries,
        logs=[]
    )
    
    final_state = workflow.invoke(initial_state)
    
    if verbose:
        print("\n".join(final_state["logs"]))
    
    return (
        final_state["sql"],
        final_state["logs"],
        final_state["is_valid"]
    )


def explain_sql_query(sql: str, question: str, schema: str) -> str:
    """
    Generate a human-readable explanation of an SQL query using Groq.
    
    Args:
        sql: The SQL query to explain
        question: The original natural language question
        schema: Database schema
    
    Returns:
        Human-readable explanation of the query
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Cannot generate explanation: GROQ_API_KEY not set"
    
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            api_key=api_key,
            max_tokens=400
        )
        
        system_prompt = """You are an SQL expert teacher. Explain SQL queries in simple, educational terms.

Your explanation should:
1. Start with a one-sentence summary of what the query does
2. Break down each part of the query (SELECT, FROM, WHERE, JOIN, GROUP BY, etc.)
3. Explain WHY each part is needed to answer the question
4. Use simple language - assume the user is learning SQL

Be concise but educational. Focus on helping users understand SQL concepts."""

        user_prompt = f"""Question: {question}

SQL Query:
{sql}

Database Schema:
{schema}

Explain this SQL query in simple terms, helping the user understand how it answers their question."""

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        return response.content.strip()
    
    except Exception as e:
        return f"Error generating explanation: {e}"


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    test_schema = """
CREATE TABLE "customers" (
    "customer_id" INTEGER,
    "name" TEXT,
    "city" TEXT,
    "region" TEXT,
    "age" INTEGER
);
"""
    
    questions = [
        "average age for each city",
        "show customers older than 30",
        "total customers by region"
    ]
    
    print("SIMPLIFIED LANGGRAPH WORKFLOW TEST")
    print("=" * 80)
    
    for q in questions:
        print(f"\nQuestion: {q}")
        print("-" * 80)
        sql, logs, valid = generate_sql_simplified(q, test_schema, verbose=False)
        print(f"SQL: {sql}")
        print(f"Valid: {valid}")
        print(f"Logs: {' → '.join(logs)}")
