"""
User Credential - Migrated from entities/user_credential.py

This module has been migrated to the new GraphBuilder enterprise structure.
Original functionality is preserved with improved organization and standards.

Migration Date: 2025-10-31
Original File: entities/user_credential.py
New Location: src/graphbuilder/domain/entities/user_credential.py
"""

class user_credential:
    uri:str
    user_name:str
    password:str
    database:str