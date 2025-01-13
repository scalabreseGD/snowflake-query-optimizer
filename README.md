# Snowflake Query Optimizer

An AI-powered tool for analyzing and optimizing Snowflake SQL queries. This application uses Claude 3.5 Sonnet to provide intelligent query analysis, optimization suggestions, and automated improvements.

## Features

### Query Analysis Modes

1. **Query History Analysis**
   - Analyze historical queries from Snowflake
   - Filter by execution time and date range
   - View performance metrics
   - Get optimization suggestions

2. **Manual Query Analysis**
   - Direct query input
   - File upload support
   - Batch analysis capabilities
   - Basic optimization suggestions

3. **Advanced Snowflake Optimization**
   - Specialized Snowflake-specific optimizations:
     - Clustering key analysis
     - Materialized view suggestions
     - Search optimization recommendations
     - Query result caching strategies
     - Partitioning advice
   - Detailed schema information input
   - Advanced configuration options
   - Organized results by category

### Core Features
- Query syntax validation and complexity scoring
- Antipattern detection
- Query categorization (Data Manipulation, Reporting, ETL, Analytics)
- Performance optimization suggestions
- LLM-powered analysis
- Advanced Snowflake-specific optimizations

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nareshshah139/snowflake-query-optimizer.git
   cd snowflake-query-optimizer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

4. Set up credentials:
   Create `.streamlit/secrets.toml` with your credentials:
   ```toml
   # Snowflake credentials
   SNOWFLAKE_ACCOUNT = "your-account"
   SNOWFLAKE_USER = "your-username"
   SNOWFLAKE_PASSWORD = "your-password"
   SNOWFLAKE_WAREHOUSE = "your-warehouse"
   SNOWFLAKE_DATABASE = "your-database"
   SNOWFLAKE_SCHEMA = "your-schema"

   # Anthropic API key
   ANTHROPIC_API_KEY = "your-anthropic-api-key"
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run src/snowflake_optimizer/app.py
   ```

2. Access the application at `http://localhost:8501`

3. Choose an analysis mode:
   - **Query History Analysis**: Analyzes your Snowflake query history
   - **Manual Query Analysis**: Input specific queries for basic analysis
   - **Advanced Optimization**: Deep analysis with Snowflake-specific optimizations

### Advanced Optimization Mode

1. Enter your SQL query
2. Configure optimization options:
   - Clustering key analysis
   - Materialization analysis
   - Search optimization
   - Caching strategy
   - Partitioning analysis

3. Add schema information (optional):
   - Table details
   - Column definitions
   - Partitioning configuration

4. View organized results:
   - Query information
   - Antipatterns
   - Optimization recommendations by category
   - Optimized query with copy functionality

## Development

- Run tests:
  ```bash
  pytest tests/ -v
  ```

- Check code quality:
  ```bash
  ruff check .
  ```

- Format code:
  ```bash
  ruff format .
  ```

## Project Structure

```
snowflake-query-optimizer/
├── src/
│   └── snowflake_optimizer/
│       ├── __init__.py
│       ├── app.py              # Streamlit application
│       ├── data_collector.py   # Snowflake query history collection
│       └── query_analyzer.py   # Query analysis and optimization
├── tests/
│   ├── __init__.py
│   └── test_query_analyzer.py
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Claude 3.5 Sonnet](https://www.anthropic.com/) 