#!/bin/bash

# Container neu bauen
docker-compose build scalping_engine

# Container neu starten
docker-compose down
docker-compose up -d

# Logs überprüfen
docker-compose logs -f scalping_engine
