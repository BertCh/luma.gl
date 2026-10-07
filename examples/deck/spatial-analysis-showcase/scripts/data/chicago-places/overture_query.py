import duckdb,sys
out=sys.argv[1]
con=duckdb.connect(); con.execute("install httpfs; load httpfs; set s3_region='us-west-2'; install spatial; load spatial")
con.execute(f"""copy (select id, names.primary as name, taxonomy.primary as cat, array_to_string(taxonomy.hierarchy,'>') as hier, basic_category, operating_status, confidence, addresses[1].freeform as addr, bbox.xmin as lon, bbox.ymin as lat
 from read_parquet('s3://overturemaps-us-west-2/release/2026-09-23.1/theme=places/type=place/*', hive_partitioning=1)
 where bbox.xmin between -87.95 and -87.52 and bbox.ymin between 41.64 and 42.03) to '{out}' (format parquet)""")
print(con.execute(f"select count(*) from '{out}'").fetchall())
