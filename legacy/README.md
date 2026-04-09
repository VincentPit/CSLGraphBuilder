# Legacy Files Directory

This directory contains files and modules from the original GraphBuilder project structure, preserved for reference and historical purposes.

## Directory Structure

- **`scripts/`** - Original main scripts (main_*.py, app.py, cli.py, demo.py, etc.)
- **`old_modules/`** - Core modules from the original structure (dbAccess.py, processing.py, llm.py, etc.)
- **`core/`**, **`database/`**, **`entities/`**, **`services/`**, **`shared/`** - Original directory structure
- **`backup_20251031_231643/`** - Backup created during migration
- **`__init__.py`** - Original root-level init file
- **`README_IMPROVED.md`** - Previous README version
- **`MIGRATION_GUIDE.md`** - Migration documentation

## Migration Status

✅ **Migration Completed Successfully**
- All functionality has been migrated to the new `src/graphbuilder/` enterprise structure
- These legacy files are kept for reference but are no longer used in the active system
- New package structure follows Domain-Driven Design principles

## New Structure

The active GraphBuilder now uses:
```
src/graphbuilder/
├── cli/                    # Command-line interface
├── core/                   # Core functionality  
├── domain/                 # Domain models and services
├── application/            # Use cases and orchestration
└── infrastructure/         # Repositories, services, config
```

## Note

These legacy files can be safely removed in future versions once you're confident the migration is complete and no additional functionality needs to be extracted.