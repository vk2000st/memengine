-- config was never part of the product design; removing from agents table
ALTER TABLE agents DROP COLUMN IF EXISTS config;
