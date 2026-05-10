-- Add email to companies for onboarding lookup
ALTER TABLE companies ADD COLUMN IF NOT EXISTS email VARCHAR(255);
CREATE UNIQUE INDEX IF NOT EXISTS ix_companies_email ON companies (email) WHERE email IS NOT NULL;
