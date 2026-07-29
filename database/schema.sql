CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    crop_name TEXT NOT NULL,
    disease_name TEXT NOT NULL,
    health_status TEXT NOT NULL CHECK (health_status IN ('Healthy', 'Diseased', 'Unknown')),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    second_prediction TEXT,
    third_prediction TEXT,
    image_quality_score REAL NOT NULL CHECK (image_quality_score BETWEEN 0 AND 100),
    inference_time_ms REAL NOT NULL CHECK (inference_time_ms >= 0),
    model_version TEXT NOT NULL,
    feedback TEXT CHECK (feedback IS NULL OR feedback IN ('Correct', 'Incorrect', 'Unsure'))
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_crop ON predictions(crop_name);

CREATE TABLE IF NOT EXISTS model_versions (
    model_version TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    training_date TEXT,
    test_accuracy REAL,
    macro_f1 REAL,
    input_size TEXT NOT NULL,
    notes TEXT
);

