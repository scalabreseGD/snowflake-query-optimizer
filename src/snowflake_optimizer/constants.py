# Define SQL antipatterns with detailed categorization
SQL_ANTIPATTERNS = {
    'PERFORMANCE': {
        'FTS001': {
            'name': 'Full Table Scan',
            'description': 'Query performs a full table scan without appropriate filtering',
            'impact': 'High',
            'detection': ['SELECT *', 'WHERE 1=1', 'no WHERE clause']
        },
        'IJN001': {
            'name': 'Inefficient Join',
            'description': 'Join conditions missing or using non-indexed columns',
            'impact': 'High',
            'detection': ['CROSS JOIN', 'cartesian product', 'missing JOIN condition']
        },
        'IDX001': {
            'name': 'Missing Index',
            'description': 'Frequently filtered or joined columns lack appropriate indexes',
            'impact': 'High',
            'detection': ['frequently filtered column', 'join key without index']
        },
        'LDT001': {
            'name': 'Large Data Transfer',
            'description': 'Query retrieves excessive data volume',
            'impact': 'High',
            'detection': ['SELECT *', 'large table without limit']
        },
        'NIN001': {
            'name': 'Not IN Subquery',
            'description': 'Using NOT IN with a subquery can lead to performance issues',
            'impact': 'Medium',
            'detection': ['NOT IN (SELECT ...)', 'subquery in NOT IN']
        },
        'ORC001': {
            'name': 'Overuse of OR Conditions',
            'description': 'Excessive OR conditions can prevent index usage',
            'impact': 'Medium',
            'detection': ['multiple OR in WHERE clause']
        },
        'IMP001': {
            'name': 'Implicit Data Type Conversion',
            'description': 'Implicit conversions can lead to index scans instead of seeks',
            'impact': 'High',
            'detection': ['data type mismatch in WHERE clause']
        },
        'NJN001': {
            'name': 'Nested Joins',
            'description': 'Deeply nested joins can complicate execution plans',
            'impact': 'Medium',
            'detection': ['multiple nested JOINs']
        },
        'SRT001': {
            'name': 'Unnecessary Sorting',
            'description': 'Sorting data unnecessarily increases query time',
            'impact': 'Low',
            'detection': ['ORDER BY without need']
        },
        'AGG001': {
            'name': 'Unnecessary Aggregation',
            'description': 'Aggregating data without necessity adds overhead',
            'impact': 'Low',
            'detection': ['GROUP BY without need']
        }
    },
    'DATA_QUALITY': {
        'NUL001': {
            'name': 'Unsafe Null Handling',
            'description': 'Improper handling of NULL values in comparisons',
            'impact': 'Medium',
            'detection': ['IS NULL', 'IS NOT NULL', 'NULL comparison']
        },
        'DTM001': {
            'name': 'Data Type Mismatch',
            'description': 'Implicit data type conversions in comparisons',
            'impact': 'Medium',
            'detection': ['implicit conversion', 'type mismatch']
        },
        'DUP001': {
            'name': 'Duplicate Rows',
            'description': 'Lack of constraints leading to duplicate data entries',
            'impact': 'High',
            'detection': ['no UNIQUE constraint', 'no PRIMARY KEY']
        },
        'UDF001': {
            'name': 'Improper Use of User-Defined Functions',
            'description': 'Using UDFs in WHERE clauses can hinder performance',
            'impact': 'Medium',
            'detection': ['UDF in WHERE clause']
        }
    },
    'COMPLEXITY': {
        'NSQ001': {
            'name': 'Nested Subquery',
            'description': 'Deeply nested subqueries that could be simplified',
            'impact': 'Medium',
            'detection': ['multiple SELECT levels', 'nested subquery']
        },
        'CJN001': {
            'name': 'Complex Join Chain',
            'description': 'Long chain of joins that could be simplified',
            'impact': 'Medium',
            'detection': ['multiple joins', 'join chain']
        },
        'CTE001': {
            'name': 'Overuse of CTEs',
            'description': 'Excessive Common Table Expressions can reduce readability',
            'impact': 'Low',
            'detection': ['multiple CTEs in query']
        }
    },
    'BEST_PRACTICE': {
        'WCD001': {
            'name': 'Wildcard Column Usage',
            'description': 'Using SELECT * instead of specific columns',
            'impact': 'Low',
            'detection': ['SELECT *']
        },
        'ALS001': {
            'name': 'Missing Table Alias',
            'description': 'Tables or subqueries without clear aliases',
            'impact': 'Low',
            'detection': ['missing AS keyword', 'no table alias']
        },
        'HNT001': {
            'name': 'Use of Hints',
            'description': 'Over-reliance on optimizer hints can reduce portability',
            'impact': 'Low',
            'detection': ['USE INDEX', 'FORCE INDEX']
        }
    },
    'SECURITY': {
        'INJ001': {
            'name': 'SQL Injection Risk',
            'description': 'Potential SQL injection vulnerabilities',
            'impact': 'High',
            'detection': ['dynamic SQL', 'string concatenation']
        },
        'PRM001': {
            'name': 'Missing Parameterization',
            'description': 'Hard-coded values instead of parameters',
            'impact': 'Medium',
            'detection': ['literal values', 'hard-coded constants']
        },
        'EXE001': {
            'name': 'Execution of Untrusted Scripts',
            'description': 'Running scripts from unverified sources',
            'impact': 'High',
            'detection': ['EXECUTE from external source']
        }
    },
    'MAINTAINABILITY': {
        'CMT001': {
            'name': 'Missing Comments',
            'description': 'Complex logic without explanatory comments',
            'impact': 'Low',
            'detection': ['complex logic', 'no comments']
        },
        'FMT001': {
            'name': 'Poor Formatting',
            'description': 'Inconsistent or poor SQL formatting',
            'impact': 'Low',
            'detection': ['inconsistent indentation', 'poor formatting']
        },
        'MAG001': {
            'name': 'Magic Numbers',
            'description': 'Use of unexplained numeric literals in queries',
            'impact': 'Low',
            'detection': ['hard-coded numbers']
        }
    }
}
