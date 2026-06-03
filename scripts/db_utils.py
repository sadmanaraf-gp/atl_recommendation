import pandas as pd
import oracledb

def read_oracle_data(query, user, password, dsn):
    """Read data from Oracle database."""
    try:
        with oracledb.connect(user=user, password=password, dsn=dsn) as connection:
            df = pd.read_sql(query, con=connection)
        return df
    except Exception as e:
        print(f"An error occurred while reading from Oracle: {e}")
        return None

def export_to_oracle(df, table_name, user, password, dsn, batch_size=500000):
    """
    Export DataFrame to an Oracle database table.
    Creates the table if it doesn't exist and inserts data in streaming batches.
    """
    def get_oracle_type(dtype):
        if pd.api.types.is_integer_dtype(dtype):
            return "NUMBER"
        elif pd.api.types.is_float_dtype(dtype):
            return "FLOAT"
        elif pd.api.types.is_bool_dtype(dtype):
            return "NUMBER(1)"
        elif pd.api.types.is_string_dtype(dtype):
            return "VARCHAR2(255)"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return "DATE"
        else:
            return "VARCHAR2(255)"

    create_table_sql = f"CREATE TABLE {table_name} ("
    for column_name, dtype in df.dtypes.items():
        oracle_type = get_oracle_type(dtype)
        create_table_sql += f'"{column_name.upper()}" {oracle_type}, '
    create_table_sql = create_table_sql.rstrip(", ") + ")"

    try:
        with oracledb.connect(user=user, password=password, dsn=dsn) as connection:
            with connection.cursor() as cursor:
                try:
                    cursor.execute(f"SELECT * FROM {table_name} WHERE 1=0")
                except oracledb.DatabaseError:
                    print(f"Table {table_name} not found. Creating it...")
                    cursor.execute(create_table_sql)
                    print(f"Table '{table_name}' created successfully.")

                print(f"Inserting data into {table_name} ({len(df):,} rows)...")
                columns = ", ".join([f'"{col.upper()}"' for col in df.columns])
                placeholders = ", ".join([f":{i+1}" for i in range(len(df.columns))])
                insert_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

                total = len(df)
                for i in range(0, total, batch_size):
                    batch_df = df.iloc[i:i + batch_size]
                    batch = [tuple(row) for row in batch_df.itertuples(index=False, name=None)]
                    cursor.executemany(insert_sql, batch)
                    connection.commit()
                    del batch
                    print(f"  Inserted {min(i + batch_size, total):,} / {total:,}")

            print(f"Data successfully exported to {table_name}.")
    except Exception as e:
        print(f"An error occurred during export: {e}")