-- Store encrypted full API key for dashboard recovery
ALTER TABLE companies ADD COLUMN IF NOT EXISTS api_key_encrypted TEXT;
