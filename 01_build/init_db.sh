#!/bin/bash

DB_NAME="citibike"
DB_USER="postgres"
DB_PASSWORD="axa_datascience"
DB_HOST="localhost"
DB_PORT="5432"

# Create the database if it doesn't exist
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -p $DB_PORT -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1 || \
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -p $DB_PORT -c "CREATE DATABASE $DB_NAME"

echo "Database '$DB_NAME' created (or already exists)."
