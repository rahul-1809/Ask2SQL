"""
Query Classifier: Determines if a natural language question is SIMPLE or COMPLEX
for intelligent routing to fine-tuned model vs Groq API.

SIMPLE queries (70-80%):
- Single table queries
- Basic aggregations (COUNT, SUM, AVG, MIN, MAX)
- Simple WHERE conditions
- Basic GROUP BY
- No JOINs or subqueries

COMPLEX queries (20-30%):
- Multiple table JOINs
- Subqueries or CTEs
- Window functions
- Advanced aggregations
- Self-joins
- Complex WHERE logic
"""

import re
from typing import Dict, Tuple


class QueryComplexity:
    """Enum-like class for query complexity levels"""
    SIMPLE = "SIMPLE"
    COMPLEX = "COMPLEX"


class QueryClassifier:
    """
    Rule-based classifier to determine query complexity.
    Uses keyword detection, schema analysis, and complexity scoring.
    """
    
    def __init__(
        self,
        complexity_threshold: int = 3,
        enable_classification: bool = True
    ):
        """
        Initialize classifier with configurable parameters.
        
        Args:
            complexity_threshold: Score >= this value → COMPLEX (default: 3)
            enable_classification: If False, always returns COMPLEX (uses Groq)
        """
        self.complexity_threshold = complexity_threshold
        self.enable_classification = enable_classification
        
        # Complexity indicators with scores
        self.complex_keywords = {
            # Multi-table operations
            'join': 2, 'inner join': 2, 'left join': 2, 'right join': 2, 
            'outer join': 2, 'cross join': 3, 'self join': 3,
            
            # Subqueries and CTEs
            'subquery': 3, 'nested': 3, 'with': 2, 'cte': 3,
            'exists': 2, 'not exists': 2, 'in (select': 3,
            
            # Window functions
            'window': 3, 'over': 3, 'partition by': 3, 'row_number': 3,
            'rank': 3, 'dense_rank': 3, 'lag': 3, 'lead': 3,
            'first_value': 3, 'last_value': 3,
            
            # Advanced operations
            'union': 2, 'intersect': 2, 'except': 2,
            'case when': 1, 'pivot': 3, 'unpivot': 3,
            
            # Temporal complexity
            'year-over-year': 3, 'yoy': 3, 'growth rate': 2,
            'running total': 2, 'cumulative': 2, 'moving average': 3,
            
            # Multiple aggregations
            'having': 1,
        }
        
        # Simple query indicators
        self.simple_keywords = {
            'total', 'sum', 'count', 'average', 'avg', 'min', 'max',
            'how many', 'how much', 'what is', 'show', 'list', 'get',
            'find', 'display', 'all', 'top', 'highest', 'lowest',
            'first', 'last', 'recent', 'latest', 'oldest'
        }
    
    def classify(self, question: str, schema: str = "") -> Tuple[str, Dict]:
        """
        Classify query complexity based on question and schema.
        
        Args:
            question: Natural language question
            schema: Database schema (used for table count analysis)
        
        Returns:
            Tuple of (complexity_level, details_dict)
            - complexity_level: "SIMPLE" or "COMPLEX"
            - details_dict: Explanation of classification decision
        """
        if not self.enable_classification:
            return QueryComplexity.COMPLEX, {
                "reason": "Classification disabled - routing to Groq API",
                "score": 0,
                "threshold": self.complexity_threshold
            }
        
        question_lower = question.lower().strip()
        complexity_score = 0
        detected_indicators = []
        
        # 1. Check for complex keywords
        for keyword, score in self.complex_keywords.items():
            if keyword in question_lower:
                complexity_score += score
                detected_indicators.append(f"{keyword} (+{score})")
        
        # 2. Analyze table count from schema
        table_count = self._count_tables_in_schema(schema)
        if table_count > 1:
            # Multiple tables likely need JOINs
            score_add = min(table_count - 1, 3)  # Cap at +3
            complexity_score += score_add
            detected_indicators.append(f"{table_count} tables (+{score_add})")
        
        # 3. Check for implicit subquery patterns
        implicit_subquery_patterns = [
            r'higher than (average|avg)',
            r'lower than (average|avg)',
            r'greater than (average|avg)',
            r'less than (average|avg)',
            r'above (average|avg)',
            r'below (average|avg)',
            r'more than (average|avg)',
            r'compared to (average|avg)'
        ]
        for pattern in implicit_subquery_patterns:
            if re.search(pattern, question_lower):
                complexity_score += 3
                detected_indicators.append(f"implicit subquery (vs average) (+3)")
                break
        
        # 4. Check for multiple aggregations
        agg_pattern = r'\b(count|sum|avg|average|min|max|total)\b'
        aggregations = re.findall(agg_pattern, question_lower)
        if len(aggregations) > 1:
            complexity_score += 1
            detected_indicators.append(f"multiple aggregations (+1)")
        
        # 4. Check for multiple aggregations
        agg_pattern = r'\b(count|sum|avg|average|min|max|total)\b'
        aggregations = re.findall(agg_pattern, question_lower)
        if len(aggregations) > 1:
            complexity_score += 1
            detected_indicators.append(f"multiple aggregations (+1)")
        
        # 5. Check for comparative/temporal queries
        comparative_patterns = [
            r'compare .* (with|to|and)',
            r'difference between',
            r'versus|vs\.',
            r'year.?over.?year|yoy',
            r'previous (year|month|quarter)',
            r'growth (rate)?',
            r'trend|change over time'
        ]
        for pattern in comparative_patterns:
            if re.search(pattern, question_lower):
                complexity_score += 2
                detected_indicators.append(f"comparative/temporal query (+2)")
                break
        
        # 6. Check question length (longer questions often more complex)
        word_count = len(question_lower.split())
        if word_count > 15:
            complexity_score += 1
            detected_indicators.append(f"long question ({word_count} words, +1)")
        
        # Determine complexity
        if complexity_score >= self.complexity_threshold:
            complexity = QueryComplexity.COMPLEX
            reason = f"Score {complexity_score} >= threshold {self.complexity_threshold}"
        else:
            complexity = QueryComplexity.SIMPLE
            reason = f"Score {complexity_score} < threshold {self.complexity_threshold}"
        
        details = {
            "complexity": complexity,
            "score": complexity_score,
            "threshold": self.complexity_threshold,
            "reason": reason,
            "indicators": detected_indicators if detected_indicators else ["No complex indicators found"],
            "table_count": table_count,
            "word_count": word_count
        }
        
        return complexity, details
    
    def _count_tables_in_schema(self, schema: str) -> int:
        """Count number of tables in schema string"""
        if not schema:
            return 1
        # Count CREATE TABLE statements
        create_table_count = len(re.findall(r'CREATE TABLE', schema, re.IGNORECASE))
        return max(create_table_count, 1)
    
    def should_use_groq(self, question: str, schema: str = "") -> bool:
        """
        Convenience method: Returns True if query should use Groq API.
        
        Args:
            question: Natural language question
            schema: Database schema
        
        Returns:
            True if COMPLEX (use Groq), False if SIMPLE (use fine-tuned model)
        """
        complexity, _ = self.classify(question, schema)
        return complexity == QueryComplexity.COMPLEX
    
    def should_use_fine_tuned(self, question: str, schema: str = "") -> bool:
        """
        Convenience method: Returns True if query should use fine-tuned model.
        
        Args:
            question: Natural language question
            schema: Database schema
        
        Returns:
            True if SIMPLE (use fine-tuned), False if COMPLEX (use Groq)
        """
        complexity, _ = self.classify(question, schema)
        return complexity == QueryComplexity.SIMPLE


# Global classifier instance (can be reconfigured)
_classifier = QueryClassifier()


def classify_query(question: str, schema: str = "") -> Tuple[str, Dict]:
    """
    Classify a query as SIMPLE or COMPLEX.
    
    Args:
        question: Natural language question
        schema: Database schema (optional)
    
    Returns:
        Tuple of (complexity_level, details_dict)
    """
    return _classifier.classify(question, schema)


def should_use_groq(question: str, schema: str = "") -> bool:
    """Returns True if query should be routed to Groq API"""
    return _classifier.should_use_groq(question, schema)


def should_use_fine_tuned(question: str, schema: str = "") -> bool:
    """Returns True if query should be routed to fine-tuned model"""
    return _classifier.should_use_fine_tuned(question, schema)


def configure_classifier(complexity_threshold: int = 3, enable: bool = True):
    """
    Reconfigure the global classifier.
    
    Args:
        complexity_threshold: Score >= this → COMPLEX (default: 3)
        enable: Enable classification (False → always use Groq)
    """
    global _classifier
    _classifier = QueryClassifier(
        complexity_threshold=complexity_threshold,
        enable_classification=enable
    )
