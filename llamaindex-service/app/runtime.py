"""Shared runtime state, initialized once in main.py's lifespan and read by
both the REST endpoints and the MCP tools."""

state: dict = {}
