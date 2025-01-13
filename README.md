# Snowflake Query Optimizer

An AI-powered tool for analyzing and optimizing Snowflake SQL queries. This application uses Claude 3.5 Sonnet to provide intelligent query analysis, optimization suggestions, and automated improvements.

## Features

- **Query Analysis**
  - Syntax validation and complexity scoring
  - Antipattern detection
  - Query categorization (Data Manipulation, Reporting, ETL, Analytics)
  - Performance optimization suggestions
  
- **Query Optimization**
  - Automated query rewriting for better performance
  - Join optimization
  - Filter improvements
  - Projection optimization
  
- **Schema Analysis**
  - Index suggestions
  - Materialization recommendations
  - Partitioning advice
  
- **Interactive UI**
  - Manual query input
  - Query history analysis
  - Real-time optimization feedback
  - Performance metrics visualization

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/snowflake-query-optimizer.git
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

3. Use the application in two modes:
   - **Query History Analysis**: Analyzes your Snowflake query history to identify optimization opportunities
   - **Manual Query Analysis**: Input specific queries for analysis and optimization

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