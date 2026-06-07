-- ARGO Oceanographic Data Schema
-- PostgreSQL optimized version

-- Create ARGO floats table
CREATE TABLE IF NOT EXISTS floats (
    float_id VARCHAR(20) PRIMARY KEY,
    model VARCHAR(50),
    platform_number VARCHAR(20),
    deploy_date TIMESTAMP,
    last_lat REAL,
    last_lon REAL,
    region VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    total_profiles INTEGER DEFAULT 0,
    last_profile_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create ARGO profiles table
CREATE TABLE IF NOT EXISTS profiles (
    profile_id VARCHAR(30) PRIMARY KEY,
    float_id VARCHAR(20) NOT NULL,
    cycle_number INTEGER,
    profile_datetime TIMESTAMP NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    n_levels INTEGER,
    summary_json TEXT,
    qc_status VARCHAR(20) DEFAULT 'good',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (float_id) REFERENCES floats(float_id)
);

-- Create measurements table (PostgreSQL with SERIAL instead of AUTOINCREMENT)
CREATE TABLE IF NOT EXISTS measurements (
    meas_id SERIAL PRIMARY KEY,
    profile_id VARCHAR(30) NOT NULL,
    depth_m REAL NOT NULL,
    temperature_c REAL,
    salinity_psu REAL,
    pressure_dbar REAL,
    oxygen_umol_kg REAL,
    chlorophyll_mg_m3 REAL,
    ph_total REAL,
    nitrate_umol_kg REAL,
    qc_flags VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (profile_id) REFERENCES profiles(profile_id)
);

-- Create metadata summaries table for vector search (PostgreSQL with SERIAL)
CREATE TABLE IF NOT EXISTS metadata_summaries (
    id SERIAL PRIMARY KEY,
    type VARCHAR(20) NOT NULL, -- 'profile', 'float', 'measurement_set'
    text_summary TEXT NOT NULL,
    ref_id VARCHAR(30) NOT NULL, -- profile_id or float_id
    metadata_json TEXT, -- structured metadata
    embedding_vector TEXT, -- serialized embedding
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_profiles_datetime ON profiles(profile_datetime);
CREATE INDEX IF NOT EXISTS idx_profiles_lat_lon ON profiles(lat, lon);
CREATE INDEX IF NOT EXISTS idx_profiles_float_id ON profiles(float_id);
CREATE INDEX IF NOT EXISTS idx_measurements_profile_id ON measurements(profile_id);
CREATE INDEX IF NOT EXISTS idx_measurements_depth ON measurements(depth_m);
CREATE INDEX IF NOT EXISTS idx_floats_region ON floats(region);
CREATE INDEX IF NOT EXISTS idx_floats_status ON floats(status);
CREATE INDEX IF NOT EXISTS idx_summaries_type_ref ON metadata_summaries(type, ref_id);

-- Views for common queries
CREATE OR REPLACE VIEW profile_summaries AS
SELECT 
    p.profile_id,
    p.float_id,
    p.profile_datetime,
    p.lat,
    p.lon,
    p.n_levels,
    f.region,
    f.status as float_status,
    COUNT(m.meas_id) as measurement_count,
    MIN(m.depth_m) as min_depth,
    MAX(m.depth_m) as max_depth,
    AVG(m.temperature_c) as avg_temperature,
    AVG(m.salinity_psu) as avg_salinity
FROM profiles p
LEFT JOIN floats f ON p.float_id = f.float_id
LEFT JOIN measurements m ON p.profile_id = m.profile_id
GROUP BY p.profile_id, p.float_id, p.profile_datetime, p.lat, p.lon, p.n_levels, f.region, f.status;

-- Create function for calculating distance (PostgreSQL version)
-- This replaces the SQLite approximation with proper PostgreSQL functions
CREATE OR REPLACE FUNCTION calculate_distance(lat1 REAL, lon1 REAL, lat2 REAL, lon2 REAL) 
RETURNS REAL AS $$
BEGIN
    RETURN 6371 * acos(
        cos(radians(lat1)) * cos(radians(lat2)) * 
        cos(radians(lon2) - radians(lon1)) + 
        sin(radians(lat1)) * sin(radians(lat2))
    );
END;
$$ LANGUAGE plpgsql;