import os
import asyncio
import subprocess
from fastapi import FastAPI, Depends, Query, BackgroundTasks, Path, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import uvicorn
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from scalping_engine.utils.config import get_settings, Settings
from scalping_engine.utils.logger import setup_logging
from scalping_engine.utils.db import init_db, get_db, get_async_db
from scalping_engine.data_manager.fetcher import fetch_market_data
from scalping_engine.data_manager.storage import DataStorage
from scalping_engine.backtesting.engine import run_backtest as run_backtest_func
from scalping_engine.backtesting.engine import BacktestConfig

# Bestehende App-Definition
app = FastAPI(
    title="Scalping Engine API",
    description="API für das Hochperformante Modulare Scalping-System",
    version="0.1.0",
)

# CORS Middleware einrichten
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion einschränken!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Beibehaltung der vorhandenen Startup- und Shutdown-Ereignisse
@app.on_event("startup")
async def startup_event():
    """
    Initialisiert Datenbank, Logging und andere Dienste beim Start der App.
    """
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    
    logger.info("Initialisiere Scalping Engine")
    
    # Datenbank initialisieren
    try:
        await init_db()
        logger.info("Datenbankverbindung hergestellt")
    except Exception as e:
        logger.error(f"Fehler bei Datenbankinitialisierung: {e}")
        if not settings.DEBUG:
            raise
    
    # Strategien laden
    try:
        from scalping_engine.strategy_engine.strategy_base import load_strategies
        loaded_strategies = load_strategies()
        logger.info(f"Strategien geladen: {', '.join(loaded_strategies)}")
    except Exception as e:
        logger.error(f"Fehler beim Laden der Strategien: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Führt Aufräumoperationen durch, wenn die App beendet wird.
    """
    logger.info("Scalping Engine wird heruntergefahren")

# Beibehaltung aller vorhandenen Endpunkte...
# ...

# Neue Modelle für Scalping-System-Runner API-Endpunkte

class RunScalpingSystemRequest(BaseModel):
    """Anfrage-Modell für die Ausführung des Scalping-System-Runners."""
    symbol: str = Field(..., description="Symbol des zu handelnden Assets (z.B. 'BTC/USDT')")
    days: int = Field(180, description="Anzahl der Tage für historische Daten")
    output_format: str = Field("json", description="Ausgabeformat (json, text)")

class OptimizeStrategyRequest(BaseModel):
    """Anfrage-Modell für die Optimierung einer Strategie."""
    symbol: str = Field(..., description="Symbol des zu handelnden Assets")
    strategy_type: str = Field(..., description="Typ der zu optimierenden Strategie")
    days: int = Field(180, description="Anzahl der Tage für historische Daten")
    parameter_ranges: Dict[str, Any] = Field({}, description="Parameter-Bereiche für die Optimierung")
    optimization_method: str = Field("grid", description="Optimierungsmethode (grid, genetic, ml)")

# API-Endpunkte für den Scalping-System-Runner

@app.post("/run-scalping-system", tags=["Scalping System Runner"])
async def run_scalping_system(request: RunScalpingSystemRequest, background_tasks: BackgroundTasks):
    """
    Führt den Scalping-System-Runner für ein bestimmtes Symbol aus.
    
    Der Runner analysiert das Symbol, wählt geeignete Strategien aus und führt Backtests durch.
    Je nach Konfiguration wird die beste Strategie ausgewählt und optimiert.
    """
    try:
        # Asynchronen Prozess starten
        background_tasks.add_task(
            execute_scalping_runner,
            symbol=request.symbol,
            days=request.days,
            output_format=request.output_format
        )
        
        return {
            "status": "started",
            "message": f"Scalping-System-Runner für {request.symbol} gestartet",
            "job_id": f"scalping_{request.symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "estimated_completion": "Der Lauf kann je nach Datenmenge mehrere Minuten dauern."
        }
    
    except Exception as e:
        logger.error(f"Fehler beim Starten des Scalping-System-Runners: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scalping-system-status/{job_id}", tags=["Scalping System Runner"])
async def get_scalping_system_status(job_id: str):
    """
    Gibt den aktuellen Status eines Scalping-System-Runner-Jobs zurück.
    
    In einer vollständigen Implementierung würde dieser Endpunkt den Status
    aus einer Datenbank oder einem Statusdienst abrufen.
    """
    # Platzhalter-Implementierung
    # In einer vollständigen Implementierung würde hier der tatsächliche Status abgerufen
    return {
        "job_id": job_id,
        "status": "running",  # Beispielwert, sollte dynamisch sein
        "progress": 50,  # Beispielwert, sollte dynamisch sein
        "start_time": (datetime.now() - timedelta(minutes=5)).isoformat(),
        "message": "Scalping-System-Runner wird ausgeführt..."
    }

@app.post("/optimize-strategy", tags=["Scalping System Runner"])
async def optimize_strategy(request: OptimizeStrategyRequest, background_tasks: BackgroundTasks):
    """
    Startet einen Optimierungsprozess für eine bestimmte Strategie.
    
    Dieser Endpunkt führt den Scalping-System-Runner im Optimierungsmodus aus,
    um die besten Parameter für eine Strategie zu finden.
    """
    try:
        # Asynchronen Prozess starten
        background_tasks.add_task(
            execute_strategy_optimization,
            symbol=request.symbol,
            strategy_type=request.strategy_type,
            days=request.days,
            parameter_ranges=request.parameter_ranges,
            optimization_method=request.optimization_method
        )
        
        return {
            "status": "started",
            "message": f"Strategieoptimierung für {request.strategy_type} auf {request.symbol} gestartet",
            "job_id": f"opt_{request.strategy_type}_{request.symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "optimization_method": request.optimization_method,
            "estimated_completion": "Die Optimierung kann je nach Parameterbereichen mehrere Stunden dauern."
        }
    
    except Exception as e:
        logger.error(f"Fehler beim Starten der Strategieoptimierung: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimization-results/{job_id}", tags=["Scalping System Runner"])
async def get_optimization_results(job_id: str):
    """
    Gibt die Ergebnisse eines Optimierungsprozesses zurück.
    
    In einer vollständigen Implementierung würde dieser Endpunkt die Ergebnisse
    aus einer Datenbank oder einem Ergebnisdienst abrufen.
    """
    # Platzhalter-Implementierung
    # In einer vollständigen Implementierung würde hier das tatsächliche Ergebnis abgerufen
    return {
        "job_id": job_id,
        "status": "completed",  # Beispielwert, sollte dynamisch sein
        "best_parameters": {  # Beispielwerte, sollten dynamisch sein
            "window": 20,
            "std_dev": 2.1,
            "rsi_period": 14
        },
        "performance_metrics": {  # Beispielwerte, sollten dynamisch sein
            "win_rate": 62.5,
            "profit_factor": 1.8,
            "max_drawdown_pct": 12.3
        },
        "completion_time": datetime.now().isoformat()
    }

# Hilfsfunktionen für die Ausführung des Scalping-System-Runners

async def execute_scalping_runner(symbol: str, days: int, output_format: str = "json"):
    """
    Führt den Scalping-System-Runner als separaten Prozess aus.
    
    Args:
        symbol: Symbol des zu handelnden Assets
        days: Anzahl der Tage für historische Daten
        output_format: Ausgabeformat (json, text)
    """
    try:
        # Pfad zur Runner-Datei
        runner_script = os.path.join(os.getcwd(), "scalping_engine", "scalping_system_runner.py")
        
        # Prüfen, ob die Datei existiert
        if not os.path.exists(runner_script):
            logger.error(f"Runner-Skript nicht gefunden: {runner_script}")
            return
        
        # Kommando zum Ausführen des Runners
        command = [
            "python", 
            runner_script, 
            "--symbol", symbol, 
            "--days", str(days)
        ]
        
        # Ausführung des Runners als separater Prozess
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Warten auf Abschluss und Ausgabe erfassen
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Fehler bei der Ausführung des Scalping-System-Runners: {stderr.decode()}")
        else:
            logger.info(f"Scalping-System-Runner erfolgreich ausgeführt für {symbol}")
            
            # Ausgabe verarbeiten
            if output_format == "json":
                # In einer vollständigen Implementierung würde hier die JSON-Ausgabe
                # verarbeitet und in die Datenbank oder einen Cache geschrieben
                pass
            
            # Hier könnten weitere Aktionen hinzugefügt werden, wie das Speichern von Ergebnissen
            # in der Datenbank oder das Senden von Benachrichtigungen
    
    except Exception as e:
        logger.error(f"Fehler bei der Ausführung des Scalping-System-Runners: {str(e)}")

async def execute_strategy_optimization(
    symbol: str, 
    strategy_type: str, 
    days: int, 
    parameter_ranges: Dict[str, Any],
    optimization_method: str
):
    """
    Führt eine Strategieoptimierung als separaten Prozess aus.
    
    Args:
        symbol: Symbol des zu handelnden Assets
        strategy_type: Typ der zu optimierenden Strategie
        days: Anzahl der Tage für historische Daten
        parameter_ranges: Parameter-Bereiche für die Optimierung
        optimization_method: Optimierungsmethode (grid, genetic, ml)
    """
    try:
        # Pfad zur Runner-Datei mit Optimierungsmodus
        runner_script = os.path.join(os.getcwd(), "scalping_engine", "strategy_optimizer.py")
        
        # Prüfen, ob die Datei existiert
        if not os.path.exists(runner_script):
            logger.error(f"Optimizer-Skript nicht gefunden: {runner_script}")
            return
        
        # Kommando zum Ausführen des Optimizers
        command = [
            "python", 
            runner_script, 
            "--symbol", symbol,
            "--strategy", strategy_type,
            "--days", str(days),
            "--method", optimization_method,
            "--params", str(parameter_ranges)  # In der vollständigen Implementierung JSON-Serialisierung verwenden
        ]
        
        # Ausführung des Optimizers als separater Prozess
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Warten auf Abschluss und Ausgabe erfassen
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Fehler bei der Strategieoptimierung: {stderr.decode()}")
        else:
            logger.info(f"Strategieoptimierung erfolgreich für {strategy_type} auf {symbol}")
            
            # Hier könnten weitere Aktionen hinzugefügt werden, wie das Speichern von Ergebnissen
            # in der Datenbank oder das Senden von Benachrichtigungen
    
    except Exception as e:
        logger.error(f"Fehler bei der Strategieoptimierung: {str(e)}")
