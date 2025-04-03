from sqlalchemy import create_engine, text

# Replace with your actual SQL Server connection string
engine = create_engine("mssql+pyodbc://username:password@server/database?driver=ODBC+Driver+17+for+SQL+Server")

# Create a connection
with engine.connect() as conn:
    # Step 1: Create a temporary table
    conn.execute(text("""
        CREATE TABLE #temp_users (
            id INT PRIMARY KEY,
            name NVARCHAR(100),
            email NVARCHAR(100)
        )
    """))

    # Step 2: Insert data into the temporary table
    conn.execute(text("""
        INSERT INTO #temp_users (id, name, email) VALUES
        (1, 'Alice', 'alice@example.com'),
        (2, 'Bob', 'bob@example.com')
    """))
    
    # Step 3: Query the temporary table
    result = conn.execute(text("SELECT * FROM #temp_users"))

    # Fetch and print results
    for row in result.fetchall():
        print(row)

    # Step 4: The temporary table will be dropped automatically when the connection closes.
