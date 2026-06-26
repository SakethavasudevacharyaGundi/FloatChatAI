class SQLWhitelist:

    ALLOWED_TABLES = {

        "measurements",
        "profiles",
        "floats"
    }

    ALLOWED_COLUMNS = {

        "measurements": {

            "profile_id",
            "depth_m",
            "temperature_c",
            "salinity_psu",
            "oxygen_umol_kg"
        },

        "profiles": {

            "profile_id",
            "float_id",
            "profile_datetime",
            "lat",
            "lon",
            "cycle_number"
        },

        "floats": {

            "float_id",
            "region",
            "status",
            "total_profiles"
        }
    }

    ALLOWED_FUNCTIONS = {

        "AVG",
        "MAX",
        "MIN",
        "COUNT",
        "SUM"
    }