import asyncio

from app.services.ingestion.netcdf_processor import NetCDFProcessor


async def main():

    processor = NetCDFProcessor()

    file_paths = [

        r"C:\Users\saket\OneDrive\Documents\floatchatai\backend\app\database\repositories\nodc_D1900975_339.nc",

        r"C:\Users\saket\OneDrive\Documents\floatchatai\backend\app\database\repositories\nodc_D1900975_340.nc",

        r"C:\Users\saket\OneDrive\Documents\floatchatai\backend\app\database\repositories\nodc_D1900975_341.nc",

        r"C:\Users\saket\OneDrive\Documents\floatchatai\backend\app\database\repositories\nodc_D1900979_339.nc",

        r"C:\Users\saket\OneDrive\Documents\floatchatai\backend\app\database\repositories\nodc_D1900979_341.nc"
    ]

    # Process NetCDF files
    processed_data = processor.process_multiple_files(
        file_paths
    )

    # Insert into PostgreSQL
    await processor.insert_into_postgres(
        processed_data
    )

    print("NetCDF ingestion completed")


if __name__ == "__main__":

    asyncio.run(main())