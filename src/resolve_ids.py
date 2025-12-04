"""
Normalizes and resolves IDs for subjects - 
subjects can be identified by multiple different IDs - 
(1) 00XX
(2) YY_PL_X

Subjects are also recognized by their specific scan date and time, 
and BIDS assigns a unique identifier to each ("sub-X")
This module helps to map between these different identifiers.

"""

