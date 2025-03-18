import os
import asyncio
import json
import subprocess
from fastapi import FastAPI, Depends, Query, BackgroundTasks, Path, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import uvicorn
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
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
from scalping_engine.backtesting.engine import BacktestConfig, backtest_engine
from scalping_engine.backtesting.performance import analyze_backtest, compare_strategies
from scalping_engine.strategy_engine.strategy_base import StrategyBase, load_strategies

# App definition
app = FastAPI(
    title="Scalping Engine API",
    description="API for the High-Performance Modular Scalping System",
    version="1.0.0",
)

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job tracking
active_jobs = {}

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """
    Initializes database, logging and other services at app startup.
    """
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    
    logger.info("Initializing Scalping Engine")
    
    # Initialize database
    try:
        await init_db()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        if not settings.DEBUG:
            raise
    
    # Load strategies
    try:
        loaded_strategies = load_strategies()
        logger.info(f"Loaded strategies: {', '.join(loaded_strategies)}")
    except Exception as e:
        logger.error(f"Error loading strategies: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Performs cleanup operations when the app is shut down.
    """
    logger.info("Shutting down Scalping Engine")

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# ========== Data Management Endpoints ==========

class DataFetchRequest(BaseModel):
    """Request model for fetching market data."""
    symbol: str = Field(..., description="Trading symbol (e.g., 'BTC/USDT')")
    timeframe: str = Field("1h", description="Timeframe (e.g., '1m', '5m', '1h', '1d')")
    start_date: Optional[datetime] = Field(None, description="Start date")
    end_date: Optional[datetime] = Field(None, description="End date")
    exchange: str = Field("binance", description="Exchange name")

@app.post("/data/fetch", tags=["Data Management"])
async def fetch_data(request: DataFetchRequest, background_tasks: BackgroundTasks):
    """
    Fetch market data for a symbol and store it in the database.
    """
    job_id = f"fetch_{request.symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Store job details
    active_jobs[job_id] = {
        "type": "data_fetch",
        "status": "running",
        "request": request.dict(),
        "start_time": datetime.now().isoformat(),
        "progress": 0
    }
    
    # Run data fetching in the background
    background_tasks.add_task(
        _fetch_and_store_data,
        job_id=job_id,
        symbol=request.symbol,
        timeframe=request.timeframe,
        start_date=request.start_date,
        end_date=request.end_date,
        exchange=request.exchange
    )
    
    return {
        "status": "started",
        "message": f"Data fetch for {request.symbol} started",
        "job_id": job_id
    }

async def _fetch_and_store_data(
    job_id: str,
    symbol: str,
    timeframe: str,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    exchange: str
):
    """Background task for fetching and storing market data."""
    try:
        # Default dates if not provided
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=180)  # 6 months by default
        
        active_jobs[job_id]["progress"] = 10
        active_jobs[job_id]["message"] = "Fetching data from exchange..."
        
        # Fetch data
        df = await fetch_market_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            exchange=exchange
        )
        
        if df.empty:
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["message"] = f"No data found for {symbol} in timeframe {timeframe}"
            return
        
        active_jobs[job_id]["progress"] = 50
        active_jobs[job_id]["message"] = f"Data fetched: {len(df)} data points. Storing in database..."
        
        # Store data using DataStorage
        storage = DataStorage()
        db_gen = get_async_db()
        db = await anext(db_gen)
        
        stored = await storage.store_dataframe(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            source=exchange,
            replace_existing=True,
            db=db
        )
        
        await db.close()
        
        if stored:
            active_jobs[job_id]["status"] = "completed"
            active_jobs[job_id]["message"] = f"Data fetch completed: {len(df)} data points stored"
            active_jobs[job_id]["progress"] = 100
            active_jobs[job_id]["completion_time"] = datetime.now().isoformat()
            active_jobs[job_id]["result"] = {
                "data_points": len(df),
                "start_date": df.index[0].isoformat() if not df.empty else None,
                "end_date": df.index[-1].isoformat() if not df.empty else None
            }
        else:
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["message"] = "Failed to store data in database"
    
    except Exception as e:
        logger.error(f"Error in data fetch job {job_id}: {str(e)}")
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["message"] = f"Error: {str(e)}"

@app.get("/data/available-symbols", tags=["Data Management"])
async def get_available_symbols():
    """
    Get a list of all available symbols in the database.
    """
    try:
        db_gen = get_async_db()
        db = await anext(db_gen)
        
        symbols = await DataStorage.get_available_symbols(db)
        
        await db.close()
        
        return {"symbols": symbols}
    
    except Exception as e:
        logger.error(f"Error getting available symbols: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/timeframes/{symbol}", tags=["Data Management"])
async def get_available_timeframes(symbol: str):
    """
    Get available timeframes for a specific symbol.
    """
    try:
        db_gen = get_async_db()
        db = await anext(db_gen)
        
        timeframes = await DataStorage.get_available_timeframes(symbol, db)
        
        await db.close()
        
        return {"symbol": symbol, "timeframes": timeframes}
    
    except Exception as e:
        logger.error(f"Error getting timeframes for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/statistics/{symbol}/{timeframe}", tags=["Data Management"])
async def get_data_statistics(symbol: str, timeframe: str):
    """
    Get statistics about available data for a symbol and timeframe.
    """
    try:
        db_gen = get_async_db()
        db = await anext(db_gen)
        
        stats = await DataStorage.get_data_statistics(symbol, timeframe, db)
        
        await db.close()
        
        return stats
    
    except Exception as e:
        logger.error(f"Error getting statistics for {symbol}/{timeframe}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== Strategy Management Endpoints ==========

@app.get("/strategies", tags=["Strategy Management"])
async def get_available_strategies():
    """
    Get a list of all available trading strategies.
    """
    try:
        # Ensure strategies are loaded
        load_strategies()
        
        strategies = StrategyBase.list_available_strategies()
        
        return {"strategies": strategies}
    
    except Exception as e:
        logger.error(f"Error getting available strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class StrategyParamsRequest(BaseModel):
    """Request model for getting strategy parameters."""
    strategy_type: str = Field(..., description="Strategy type")

@app.post("/strategies/params", tags=["Strategy Management"])
async def get_strategy_parameters(request: StrategyParamsRequest):
    """
    Get the parameters for a specific strategy.
    """
    try:
        # Ensure strategies are loaded
        load_strategies()
        
        # Get strategy class
        strategy_class = StrategyBase.get_strategy_class(request.strategy_type)
        
        # Create a default instance to get the default parameters
        instance = strategy_class()
        
        return {
            "strategy_type": request.strategy_type,
            "parameters": instance.parameters
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting strategy parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== Backtesting Endpoints ==========

class BacktestRequest(BaseModel):
    """Request model for running a backtest."""
    strategy_type: str = Field(..., description="Strategy type to backtest")
    strategy_params: Dict[str, Any] = Field({}, description="Strategy parameters")
    symbol: str = Field(..., description="Symbol to backtest")
    timeframe: str = Field("1h", description="Timeframe for backtesting")
    start_date: Optional[datetime] = Field(None, description="Start date for backtest")
    end_date: Optional[datetime] = Field(None, description="End date for backtest")
    initial_capital: float = Field(10000.0, description="Initial capital")
    commission: float = Field(0.001, description="Commission rate (e.g., 0.001 for 0.1%)")
    risk_per_trade: float = Field(1.0, description="Risk per trade in % of capital")

@app.post("/backtest/run", tags=["Backtesting"])
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Run a backtest for a specific strategy and symbol.
    """
    job_id = f"backtest_{request.strategy_type}_{request.symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Store job details
    active_jobs[job_id] = {
        "type": "backtest",
        "status": "running",
        "request": request.dict(),
        "start_time": datetime.now().isoformat(),
        "progress": 0
    }
    
    # Run backtest in the background
    background_tasks.add_task(
        _run_backtest_task,
        job_id=job_id,
        strategy_type=request.strategy_type,
        strategy_params=request.strategy_params,
        symbol=request.symbol,
        timeframe=request.timeframe,
        start_date=request.start_date,
        end_date=request.end_date,
        initial_capital=request.initial_capital,
        commission=request.commission,
        risk_per_trade=request.risk_per_trade
    )
    
    return {
        "status": "started",
        "message": f"Backtest for {request.strategy_type} on {request.symbol} started",
        "job_id": job_id
    }

async def _run_backtest_task(
    job_id: str,
    strategy_type: str,
    strategy_params: Dict[str, Any],
    symbol: str,
    timeframe: str,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    initial_capital: float,
    commission: float,
    risk_per_trade: float
):
    """Background task for running a backtest."""
    try:
        active_jobs[job_id]["progress"] = 10
        active_jobs[job_id]["message"] = "Creating backtest configuration..."
        
        # Create a database session
        db_gen = get_async_db()
        db = await anext(db_gen)
        
        active_jobs[job_id]["progress"] = 20
        active_jobs[job_id]["message"] = "Running backtest..."
        
        # Run the backtest
        result = await run_backtest_func(
            strategy_type=strategy_type,
            strategy_params=strategy_params,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            commission=commission,
            risk_per_trade=risk_per_trade,
            save_results=True,
            db=db
        )
        
        await db.close()
        
        if "error" in result:
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["message"] = result["error"]
            return
        
        active_jobs[job_id]["progress"] = 90
        active_jobs[job_id]["message"] = "Backtest completed, analyzing results..."
        
        # Store the results
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["message"] = "Backtest completed successfully"
        active_jobs[job_id]["progress"] = 100
        active_jobs[job_id]["completion_time"] = datetime.now().isoformat()
        active_jobs[job_id]["result"] = {
            "summary": {
                "total_profit_loss_pct": result.get("total_profit_loss_pct", 0),
                "max_drawdown_pct": result.get("max_drawdown_pct", 0),
                "win_rate": result.get("win_rate", 0),
                "profit_factor": result.get("profit_factor", 0),
                "num_trades": result.get("num_trades", 0),
                "sharpe_ratio": result.get("sharpe_ratio", 0)
            },
            "backtest_id": result.get("metrics", {}).get("backtest_id", None),
            "file_path": result.get("file_path", None)
        }
    
    except Exception as e:
        logger.error(f"Error in backtest job {job_id}: {str(e)}")
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["message"] = f"Error: {str(e)}"

@app.get("/backtest/{backtest_id}", tags=["Backtesting"])
@app.get("/job/{job_id}", tags=["Jobs"])
async def get_job_status(job_id: str):
    """
    Get the status of a job (backtest, optimization, scalping system).
    """
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
    
    return active_jobs[job_id]

@app.get("/jobs", tags=["Jobs"])
async def list_jobs(limit: int = 10, status: str = None):
    """
    List all jobs with optional filtering by status.
    """
    result = []
    
    for job_id, job_data in active_jobs.items():
        if status is None or job_data.get("status") == status:
            # Create a summary version of the job to avoid sending too much data
            job_summary = {
                "job_id": job_id,
                "type": job_data.get("type"),
                "status": job_data.get("status"),
                "start_time": job_data.get("start_time"),
                "completion_time": job_data.get("completion_time", None),
                "progress": job_data.get("progress", 0),
                "message": job_data.get("message", "")
            }
            
            if "request" in job_data:
                job_summary["request_summary"] = {
                    k: v for k, v in job_data["request"].items() 
                    if k in ["symbol", "strategy_type", "timeframe", "optimization_method"]
                }
            
            result.append(job_summary)
    
    # Sort by start time (most recent first)
    result.sort(key=lambda x: x.get("start_time", ""), reverse=True)
    
    # Apply limit
    result = result[:limit]
    
    return {"jobs": result, "count": len(result), "total": len(active_jobs)}

@app.get("/backtest/{backtest_id}", tags=["Backtesting"])
async def get_backtest_results(backtest_id: int):
    """
    Get the results of a completed backtest by ID.
    """
    try:
        # Get backtest data from database
        backtest_data = await backtest_engine.load_backtest(backtest_id)
        
        if not backtest_data:
            raise HTTPException(status_code=404, detail=f"Backtest with ID {backtest_id} not found")
        
        return {
            "id": backtest_id,
            "strategy": backtest_data.get("strategy", {}),
            "symbol": backtest_data.get("symbol", ""),
            "timeframe": backtest_data.get("timeframe", ""),
            "results": backtest_data.get("results", {}),
            "metrics": backtest_data.get("metrics", {})
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting backtest results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtest/{backtest_id}/analyze", tags=["Backtesting"])
async def analyze_backtest_results(backtest_id: int, save_report: bool = True):
    """
    Analyze a backtest's results and optionally create a performance report.
    """
    try:
        # Analyze backtest
        analysis = await analyze_backtest(backtest_id, save_report)
        
        if "error" in analysis:
            raise HTTPException(status_code=404, detail=analysis["error"])
        
        return analysis
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== Scalping System Runner Endpoints ==========

class RunScalpingSystemRequest(BaseModel):
    """Request model for running the scalping system."""
    symbol: str = Field(..., description="Symbol to trade (e.g., 'BTC/USDT')")
    days: int = Field(180, description="Number of days for historical data")
    output_format: str = Field("json", description="Output format (json, text)")

@app.post("/run-scalping-system", tags=["Scalping System Runner"])
async def run_scalping_system(request: RunScalpingSystemRequest, background_tasks: BackgroundTasks):
    """
    Run the scalping system runner for a specific symbol.
    
    The runner analyzes the symbol, selects appropriate strategies, and runs backtests.
    """
    job_id = f"scalping_{request.symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Store job details
    active_jobs[job_id] = {
        "type": "scalping_system",
        "status": "running",
        "request": request.dict(),
        "start_time": datetime.now().isoformat(),
        "progress": 0
    }
    
    # Run the scalping system runner in the background
    background_tasks.add_task(
        execute_scalping_runner,
        job_id=job_id,
        symbol=request.symbol,
        days=request.days,
        output_format=request.output_format
    )
    
    return {
        "status": "started",
        "message": f"Scalping System Runner for {request.symbol} started",
        "job_id": job_id,
        "estimated_completion": "The process may take several minutes depending on the amount of data."
    }

async def execute_scalping_runner(job_id: str, symbol: str, days: int, output_format: str = "json"):
    """
    Executes the scalping system runner as a background process.
    
    Args:
        job_id: ID of the job
        symbol: Symbol to trade
        days: Number of days for historical data
        output_format: Output format (json, text)
    """
    try:
        active_jobs[job_id]["message"] = "Starting scalping system analysis..."
        active_jobs[job_id]["progress"] = 5
        
        # Use Python's asyncio subprocess to run the script
        runner_script = os.path.join(os.getcwd(), "scalping_system_runner.py")
        
        if not os.path.exists(runner_script):
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["message"] = f"Runner script not found: {runner_script}"
            return
        
        active_jobs[job_id]["message"] = f"Analyzing {symbol}..."
        active_jobs[job_id]["progress"] = 10
        
        # Command to run the runner
        cmd = [
            "python", 
            runner_script, 
            "--symbol", symbol, 
            "--days", str(days)
        ]
        
        # Create process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        active_jobs[job_id]["message"] = "Scalping system runner is executing..."
        active_jobs[job_id]["progress"] = 30
        
        # Wait for completion and capture output
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode()
            logger.error(f"Error in scalping system runner: {error_msg}")
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["message"] = f"Error in scalping system runner: {error_msg[:500]}..."
            return
        
        output = stdout.decode()
        
        # Process output based on format
        if output_format == "json":
            # Try to parse JSON from the output
            try:
                # Extract JSON parts from the output
                import re
                json_blocks = re.findall(r'({.*?})', output, re.DOTALL)
                parsed_results = []
                
                for block in json_blocks:
                    try:
                        parsed_block = json.loads(block)
                        parsed_results.append(parsed_block)
                    except:
                        pass
                
                result_data = {
                    "output": output,
                    "parsed_results": parsed_results
                }
            except Exception as e:
                logger.warning(f"Could not parse JSON from output: {str(e)}")
                result_data = {"output": output}
        else:
            result_data = {"output": output}
        
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["message"] = f"Scalping system analysis completed for {symbol}"
        active_jobs[job_id]["progress"] = 100
        active_jobs[job_id]["completion_time"] = datetime.now().isoformat()
        active_jobs[job_id]["result"] = result_data
        
        logger.info(f"Completed scalping system job {job_id} for {symbol}")
    
    except Exception as e:
        logger.error(f"Error in scalping system job {job_id}: {str(e)}")
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["message"] = f"Error: {str(e)}"

class OptimizeStrategyRequest(BaseModel):
    """Request model for optimizing a strategy."""
    symbol: str = Field(..., description="Symbol to trade")
    strategy_type: str = Field(..., description="Strategy type to optimize")
    days: int = Field(180, description="Number of days for historical data")
    parameter_ranges: Dict[str, Any] = Field({}, description="Parameter ranges for optimization")
    optimization_method: str = Field("grid", description="Optimization method (grid, genetic, ml)")

@app.post("/optimize-strategy", tags=["Scalping System Runner"])
async def optimize_strategy(request: OptimizeStrategyRequest, background_tasks: BackgroundTasks):
    """
    Start an optimization process for a specific strategy.
    """
    job_id = f"opt_{request.strategy_type}_{request.symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Store job details
    active_jobs[job_id] = {
        "type": "strategy_optimization",
        "status": "running",
        "request": request.dict(),
        "start_time": datetime.now().isoformat(),
        "progress": 0
    }
    
    # Run strategy optimization in the background
    background_tasks.add_task(
        execute_strategy_optimization,
        job_id=job_id,
        symbol=request.symbol,
        strategy_type=request.strategy_type,
        days=request.days,
        parameter_ranges=request.parameter_ranges,
        optimization_method=request.optimization_method
    )
    
    return {
        "status": "started",
        "message": f"Strategy optimization for {request.strategy_type} on {request.symbol} started",
        "job_id": job_id,
        "optimization_method": request.optimization_method,
        "estimated_completion": "Optimization may take several hours depending on parameter ranges."
    }

async def execute_strategy_optimization(
    job_id: str,
    symbol: str,
    strategy_type: str,
    days: int,
    parameter_ranges: Dict[str, Any],
    optimization_method: str
):
    """
    Executes strategy optimization as a background process.
    
    Args:
        job_id: ID of the job
        symbol: Symbol to trade
        strategy_type: Type of strategy to optimize
        days: Number of days for historical data
        parameter_ranges: Parameter ranges for optimization
        optimization_method: Optimization method (grid, genetic, ml)
    """
    try:
        active_jobs[job_id]["message"] = "Starting strategy optimization..."
        active_jobs[job_id]["progress"] = 5
        
        # Use Python's asyncio subprocess to run the optimizer
        optimizer_script = os.path.join(os.getcwd(), "strategy_optimizer.py")
        
        # Check if script exists
        if not os.path.exists(optimizer_script):
            # If not, we'll implement the optimization here directly
            active_jobs[job_id]["message"] = "Strategy optimizer script not found. Using internal optimization..."
            await _perform_internal_optimization(
                job_id=job_id,
                symbol=symbol,
                strategy_type=strategy_type,
                days=days,
                parameter_ranges=parameter_ranges,
                optimization_method=optimization_method
            )
            return
        
        active_jobs[job_id]["message"] = f"Optimizing {strategy_type} for {symbol}..."
        active_jobs[job_id]["progress"] = 10
        
        # Command to run the optimizer
        cmd = [
            "python", 
            optimizer_script, 
            "--symbol", symbol,
            "--strategy", strategy_type,
            "--days", str(days),
            "--method", optimization_method,
            "--params", json.dumps(parameter_ranges)
        ]
        
        # Create process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        active_jobs[job_id]["message"] = "Optimization is running..."
        active_jobs[job_id]["progress"] = 30
        
        # Wait for completion and capture output
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode()
            logger.error(f"Error in strategy optimization: {error_msg}")
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["message"] = f"Error in strategy optimization: {error_msg[:500]}..."
            return
        
        output = stdout.decode()
        
        # Try to parse JSON results
        try:
            import re
            json_blocks = re.findall(r'({.*?})', output, re.DOTALL)
            parsed_results = []
            
            for block in json_blocks:
                try:
                    parsed_block = json.loads(block)
                    parsed_results.append(parsed_block)
                except:
                    pass
            
            # Find the best parameters
            best_params = None
            best_performance = -float('inf')
            
            for result in parsed_results:
                if 'performance' in result and 'parameters' in result:
                    if result['performance'] > best_performance:
                        best_performance = result['performance']
                        best_params = result['parameters']
            
            result_data = {
                "output": output,
                "parsed_results": parsed_results,
                "best_parameters": best_params,
                "best_performance": best_performance
            }
        except Exception as e:
            logger.warning(f"Could not parse JSON from output: {str(e)}")
            result_data = {"output": output}
        
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["message"] = f"Strategy optimization completed for {strategy_type} on {symbol}"
        active_jobs[job_id]["progress"] = 100
        active_jobs[job_id]["completion_time"] = datetime.now().isoformat()
        active_jobs[job_id]["result"] = result_data
        
        logger.info(f"Completed strategy optimization job {job_id} for {strategy_type} on {symbol}")
    
    except Exception as e:
        logger.error(f"Error in strategy optimization job {job_id}: {str(e)}")
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["message"] = f"Error: {str(e)}"

async def _perform_internal_optimization(
    job_id: str,
    symbol: str,
    strategy_type: str,
    days: int,
    parameter_ranges: Dict[str, Any],
    optimization_method: str
):
    """
    Performs strategy optimization internally (without using an external script).
    
    Args:
        job_id: ID of the job
        symbol: Symbol to trade
        strategy_type: Type of strategy to optimize
        days: Number of days for historical data
        parameter_ranges: Parameter ranges for optimization
        optimization_method: Optimization method (grid, genetic, ml)
    """
    try:
        active_jobs[job_id]["message"] = "Starting internal optimization process..."
        active_jobs[job_id]["progress"] = 15
        
        # Ensure strategies are loaded
        load_strategies()
        
        # Prepare data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get market data
        active_jobs[job_id]["message"] = f"Fetching market data for {symbol}..."
        active_jobs[job_id]["progress"] = 20
        
        data = await fetch_market_data(
            symbol=symbol,
            timeframe="1h",  # Default timeframe for optimization
            start_date=start_date,
            end_date=end_date
        )
        
        if data.empty:
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["message"] = f"No data found for {symbol} in the specified date range"
            return
            
        active_jobs[job_id]["message"] = f"Data fetched: {len(data)} data points. Starting parameter optimization..."
        active_jobs[job_id]["progress"] = 30
        
        # Initialize results storage
        optimization_results = []
        
        if optimization_method == "grid":
            # Prepare parameter grid
            param_grid = []
            
            for param, value_range in parameter_ranges.items():
                if isinstance(value_range, list):
                    # Already a list of values
                    values = value_range
                elif isinstance(value_range, dict) and all(k in value_range for k in ['min', 'max', 'step']):
                    # Range specification
                    min_val = value_range['min']
                    max_val = value_range['max']
                    step = value_range['step']
                    
                    if isinstance(min_val, int) and isinstance(max_val, int) and isinstance(step, int):
                        values = list(range(min_val, max_val + 1, step))
                    else:
                        # Floating point range
                        values = []
                        current = min_val
                        while current <= max_val:
                            values.append(current)
                            current += step
                else:
                    # Single value
                    values = [value_range]
                
                param_grid.append({'param': param, 'values': values})
            
            # Generate all combinations of parameters
            import itertools
            param_combinations = []
            
            # Get all parameter names and their respective values
            param_names = [item['param'] for item in param_grid]
            param_values = [item['values'] for item in param_grid]
            
            # Generate all combinations
            for combo_values in itertools.product(*param_values):
                param_combo = dict(zip(param_names, combo_values))
                param_combinations.append(param_combo)
            
            total_combinations = len(param_combinations)
            active_jobs[job_id]["message"] = f"Generated {total_combinations} parameter combinations for grid search"
            
            # Run backtest for each parameter combination
            for i, params in enumerate(param_combinations):
                progress = 30 + (i / total_combinations) * 60
                active_jobs[job_id]["progress"] = int(progress)
                active_jobs[job_id]["message"] = f"Testing combination {i+1}/{total_combinations}: {params}"
                
                # Create backtest config
                config = BacktestConfig(
                    strategy_type=strategy_type,
                    strategy_params=params,
                    symbol=symbol,
                    timeframe="1h",
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=10000.0
                )
                
                # Run backtest
                results, metrics, _ = await backtest_engine.run_backtest(config)
                
                if not metrics:
                    logger.warning(f"No metrics returned for parameter combination: {params}")
                    continue
                
                # Calculate performance score
                # Weighted combination of key metrics
                win_rate = metrics.get('win_rate', 0)
                profit_factor = metrics.get('profit_factor', 0)
                total_profit_pct = metrics.get('total_profit_loss_pct', 0)
                max_dd_pct = metrics.get('max_drawdown_pct', 0) or 1  # Avoid division by zero
                
                # Performance score formula
                score = (win_rate * 0.3) + (profit_factor * 20 * 0.3) + (total_profit_pct * 0.3) - (max_dd_pct * 0.1)
                
                # Save result
                optimization_results.append({
                    'parameters': params,
                    'metrics': {
                        'win_rate': win_rate,
                        'profit_factor': profit_factor,
                        'total_profit_loss_pct': total_profit_pct,
                        'max_drawdown_pct': max_dd_pct,
                        'num_trades': metrics.get('num_trades', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0)
                    },
                    'performance_score': score
                })
                
                logger.info(f"Combination {i+1}/{total_combinations} completed. Score: {score:.2f}")
        
        elif optimization_method == "genetic":
            # Simplified genetic algorithm implementation
            active_jobs[job_id]["message"] = "Using genetic algorithm optimization..."
            
            # Initial population size
            population_size = 20
            generations = 5
            mutation_rate = 0.2
            
            # Generate initial population
            population = []
            
            for _ in range(population_size):
                individual = {}
                for param, value_range in parameter_ranges.items():
                    if isinstance(value_range, list):
                        individual[param] = np.random.choice(value_range)
                    elif isinstance(value_range, dict) and all(k in value_range for k in ['min', 'max']):
                        min_val = value_range['min']
                        max_val = value_range['max']
                        
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            individual[param] = np.random.randint(min_val, max_val + 1)
                        else:
                            individual[param] = np.random.uniform(min_val, max_val)
                    else:
                        individual[param] = value_range
                
                population.append(individual)
            
            # Evaluate initial population
            fitnesses = []
            
            for i, params in enumerate(population):
                progress = 30 + (i / (population_size * generations)) * 60
                active_jobs[job_id]["progress"] = int(progress)
                active_jobs[job_id]["message"] = f"Evaluating individual {i+1}/{population_size} in generation 1"
                
                # Create backtest config
                config = BacktestConfig(
                    strategy_type=strategy_type,
                    strategy_params=params,
                    symbol=symbol,
                    timeframe="1h",
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=10000.0
                )
                
                # Run backtest
                results, metrics, _ = await backtest_engine.run_backtest(config)
                
                if not metrics:
                    fitnesses.append(0)
                    continue
                
                # Calculate fitness
                win_rate = metrics.get('win_rate', 0)
                profit_factor = metrics.get('profit_factor', 0)
                total_profit_pct = metrics.get('total_profit_loss_pct', 0)
                max_dd_pct = metrics.get('max_drawdown_pct', 0) or 1
                
                # Fitness score
                score = (win_rate * 0.3) + (profit_factor * 20 * 0.3) + (total_profit_pct * 0.3) - (max_dd_pct * 0.1)
                fitnesses.append(score)
                
                # Save result
                optimization_results.append({
                    'parameters': params,
                    'metrics': {
                        'win_rate': win_rate,
                        'profit_factor': profit_factor,
                        'total_profit_loss_pct': total_profit_pct,
                        'max_drawdown_pct': max_dd_pct,
                        'num_trades': metrics.get('num_trades', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0)
                    },
                    'performance_score': score,
                    'generation': 1
                })
            
            # Evolution loop
            for generation in range(2, generations + 1):
                # Select parents
                parent_indices = np.argsort(fitnesses)[-int(population_size/2):]  # Select best half
                parents = [population[i] for i in parent_indices]
                parent_fitnesses = [fitnesses[i] for i in parent_indices]
                
                # Create next generation
                next_population = []
                next_population.extend(parents)  # Elitism - keep best parents
                
                # Create children through crossover and mutation
                while len(next_population) < population_size:
                    # Select two parents weighted by fitness
                    parent1, parent2 = np.random.choice(
                        parents, 
                        size=2, 
                        p=np.array(parent_fitnesses) / sum(parent_fitnesses)
                    )
                    
                    # Crossover
                    child = {}
                    for param in parent1.keys():
                        # 50% chance to inherit from each parent
                        if np.random.random() < 0.5:
                            child[param] = parent1[param]
                        else:
                            child[param] = parent2[param]
                    
                    # Mutation
                    for param, value_range in parameter_ranges.items():
                        if np.random.random() < mutation_rate:
                            if isinstance(value_range, list):
                                child[param] = np.random.choice(value_range)
                            elif isinstance(value_range, dict) and all(k in value_range for k in ['min', 'max']):
                                min_val = value_range['min']
                                max_val = value_range['max']
                                
                                if isinstance(min_val, int) and isinstance(max_val, int):
                                    child[param] = np.random.randint(min_val, max_val + 1)
                                else:
                                    child[param] = np.random.uniform(min_val, max_val)
                    
                    next_population.append(child)
                
                # Update population
                population = next_population
                fitnesses = []
                
                # Evaluate new population
                for i, params in enumerate(population):
                    base_progress = 30 + ((generation - 1) / generations) * 60
                    progress = base_progress + (i / (population_size * generations)) * 60
                    active_jobs[job_id]["progress"] = int(progress)
                    active_jobs[job_id]["message"] = f"Evaluating individual {i+1}/{population_size} in generation {generation}"
                    
                    # Create backtest config
                    config = BacktestConfig(
                        strategy_type=strategy_type,
                        strategy_params=params,
                        symbol=symbol,
                        timeframe="1h",
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=10000.0
                    )
                    
                    # Run backtest
                    results, metrics, _ = await backtest_engine.run_backtest(config)
                    
                    if not metrics:
                        fitnesses.append(0)
                        continue
                    
                    # Calculate fitness
                    win_rate = metrics.get('win_rate', 0)
                    profit_factor = metrics.get('profit_factor', 0)
                    total_profit_pct = metrics.get('total_profit_loss_pct', 0)
                    max_dd_pct = metrics.get('max_drawdown_pct', 0) or 1
                    
                    # Fitness score
                    score = (win_rate * 0.3) + (profit_factor * 20 * 0.3) + (total_profit_pct * 0.3) - (max_dd_pct * 0.1)
                    fitnesses.append(score)
                    
                    # Save result
                    optimization_results.append({
                        'parameters': params,
                        'metrics': {
                            'win_rate': win_rate,
                            'profit_factor': profit_factor,
                            'total_profit_loss_pct': total_profit_pct,
                            'max_drawdown_pct': max_dd_pct,
                            'num_trades': metrics.get('num_trades', 0),
                            'sharpe_ratio': metrics.get('sharpe_ratio', 0)
                        },
                        'performance_score': score,
                        'generation': generation
                    })
        
        elif optimization_method == "ml":
            # Simple ML-based optimization
            active_jobs[job_id]["message"] = "Using ML-based optimization (randomized search with learning)..."
            
            # Number of iterations
            iterations = 30
            exploration_ratio = 0.7  # 70% exploration, 30% exploitation
            
            # Run initial random searches
            initial_samples = 10
            
            # Initialize results storage
            samples = []
            X_samples = []
            y_samples = []
            
            # Generate initial random samples
            for i in range(initial_samples):
                progress = 30 + (i / iterations) * 60
                active_jobs[job_id]["progress"] = int(progress)
                active_jobs[job_id]["message"] = f"Generating initial sample {i+1}/{initial_samples}"
                
                # Generate random parameters
                params = {}
                for param, value_range in parameter_ranges.items():
                    if isinstance(value_range, list):
                        params[param] = np.random.choice(value_range)
                    elif isinstance(value_range, dict) and all(k in value_range for k in ['min', 'max']):
                        min_val = value_range['min']
                        max_val = value_range['max']
                        
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            params[param] = np.random.randint(min_val, max_val + 1)
                        else:
                            params[param] = np.random.uniform(min_val, max_val)
                    else:
                        params[param] = value_range
                
                # Create backtest config
                config = BacktestConfig(
                    strategy_type=strategy_type,
                    strategy_params=params,
                    symbol=symbol,
                    timeframe="1h",
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=10000.0
                )
                
                # Run backtest
                results, metrics, _ = await backtest_engine.run_backtest(config)
                
                if not metrics:
                    continue
                
                # Calculate performance score
                win_rate = metrics.get('win_rate', 0)
                profit_factor = metrics.get('profit_factor', 0)
                total_profit_pct = metrics.get('total_profit_loss_pct', 0)
                max_dd_pct = metrics.get('max_drawdown_pct', 0) or 1
                
                # Performance score formula
                score = (win_rate * 0.3) + (profit_factor * 20 * 0.3) + (total_profit_pct * 0.3) - (max_dd_pct * 0.1)
                
                # Save result
                optimization_results.append({
                    'parameters': params,
                    'metrics': {
                        'win_rate': win_rate,
                        'profit_factor': profit_factor,
                        'total_profit_loss_pct': total_profit_pct,
                        'max_drawdown_pct': max_dd_pct,
                        'num_trades': metrics.get('num_trades', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0)
                    },
                    'performance_score': score,
                    'iteration': i + 1
                })
                
                # Add to samples for ML model
                samples.append((params, score))
                
                # Create feature vector from parameters
                feature_vector = []
                for param, value_range in parameter_ranges.items():
                    feature_vector.append(params.get(param, 0))
                
                X_samples.append(feature_vector)
                y_samples.append(score)
            
            # Train a simple regression model
            from sklearn.ensemble import RandomForestRegressor
            
            if len(X_samples) > 0:
                try:
                    model = RandomForestRegressor(n_estimators=10)
                    model.fit(X_samples, y_samples)
                    
                    # Use the model for guided exploration
                    for i in range(initial_samples, iterations):
                        progress = 30 + (i / iterations) * 60
                        active_jobs[job_id]["progress"] = int(progress)
                        
                        # Decide between exploration and exploitation
                        if np.random.random() < exploration_ratio:
                            # Exploration - try random parameters
                            active_jobs[job_id]["message"] = f"Exploration phase - iteration {i+1}/{iterations}"
                            params = {}
                            for param, value_range in parameter_ranges.items():
                                if isinstance(value_range, list):
                                    params[param] = np.random.choice(value_range)
                                elif isinstance(value_range, dict) and all(k in value_range for k in ['min', 'max']):
                                    min_val = value_range['min']
                                    max_val = value_range['max']
                                    
                                    if isinstance(min_val, int) and isinstance(max_val, int):
                                        params[param] = np.random.randint(min_val, max_val + 1)
                                    else:
                                        params[param] = np.random.uniform(min_val, max_val)
                                else:
                                    params[param] = value_range
                        else:
                            # Exploitation - use model to predict good parameters
                            active_jobs[job_id]["message"] = f"Exploitation phase - iteration {i+1}/{iterations}"
                            
                            # Generate multiple random candidates and pick the one with highest predicted score
                            num_candidates = 100
                            candidates = []
                            
                            for _ in range(num_candidates):
                                candidate_params = {}
                                for param, value_range in parameter_ranges.items():
                                    if isinstance(value_range, list):
                                        candidate_params[param] = np.random.choice(value_range)
                                    elif isinstance(value_range, dict) and all(k in value_range for k in ['min', 'max']):
                                        min_val = value_range['min']
                                        max_val = value_range['max']
                                        
                                        if isinstance(min_val, int) and isinstance(max_val, int):
                                            candidate_params[param] = np.random.randint(min_val, max_val + 1)
                                        else:
                                            candidate_params[param] = np.random.uniform(min_val, max_val)
                                    else:
                                        candidate_params[param] = value_range
                                
                                # Create feature vector
                                feature_vector = []
                                for param, value_range in parameter_ranges.items():
                                    feature_vector.append(candidate_params.get(param, 0))
                                
                                # Predict score
                                predicted_score = model.predict([feature_vector])[0]
                                candidates.append((candidate_params, predicted_score))
                            
                            # Select best candidate
                            candidates.sort(key=lambda x: x[1], reverse=True)
                            params = candidates[0][0]
                        
                        # Create backtest config
                        config = BacktestConfig(
                            strategy_type=strategy_type,
                            strategy_params=params,
                            symbol=symbol,
                            timeframe="1h",
                            start_date=start_date,
                            end_date=end_date,
                            initial_capital=10000.0
                        )
                        
                        # Run backtest
                        results, metrics, _ = await backtest_engine.run_backtest(config)
                        
                        if not metrics:
                            continue
                        
                        # Calculate performance score
                        win_rate = metrics.get('win_rate', 0)
                        profit_factor = metrics.get('profit_factor', 0)
                        total_profit_pct = metrics.get('total_profit_loss_pct', 0)
                        max_dd_pct = metrics.get('max_drawdown_pct', 0) or 1
                        
                        # Performance score formula
                        score = (win_rate * 0.3) + (profit_factor * 20 * 0.3) + (total_profit_pct * 0.3) - (max_dd_pct * 0.1)
                        
                        # Save result
                        optimization_results.append({
                            'parameters': params,
                            'metrics': {
                                'win_rate': win_rate,
                                'profit_factor': profit_factor,
                                'total_profit_loss_pct': total_profit_pct,
                                'max_drawdown_pct': max_dd_pct,
                                'num_trades': metrics.get('num_trades', 0),
                                'sharpe_ratio': metrics.get('sharpe_ratio', 0)
                            },
                            'performance_score': score,
                            'iteration': i + 1
                        })
                        
                        # Add to samples for ML model
                        samples.append((params, score))
                        
                        # Create feature vector from parameters
                        feature_vector = []
                        for param, value_range in parameter_ranges.items():
                            feature_vector.append(params.get(param, 0))
                        
                        X_samples.append(feature_vector)
                        y_samples.append(score)
                        
                        # Retrain model periodically
                        if (i - initial_samples) % 5 == 0:
                            model.fit(X_samples, y_samples)
                except Exception as e:
                    logger.warning(f"Error in ML optimization: {str(e)}")
            
        # Find best result
        best_result = None
        best_score = -float('inf')
        
        for result in optimization_results:
            if result['performance_score'] > best_score:
                best_score = result['performance_score']
                best_result = result
        
        # Final validation of best parameters
        if best_result:
            active_jobs[job_id]["message"] = "Running final validation with best parameters..."
            active_jobs[job_id]["progress"] = 90
            
            # Create backtest config with best parameters
            config = BacktestConfig(
                strategy_type=strategy_type,
                strategy_params=best_result['parameters'],
                symbol=symbol,
                timeframe="1h",
                start_date=start_date,
                end_date=end_date,
                initial_capital=10000.0
            )
            
            # Run final backtest
            results, metrics, trades = await backtest_engine.run_backtest(config)
            
            if metrics:
                # Update best result with final metrics
                best_result['final_validation'] = {
                    'metrics': metrics,
                    'trades_count': len(trades) if trades else 0
                }
        
        # Generate optimization report
        active_jobs[job_id]["message"] = "Optimization completed, generating report..."
        active_jobs[job_id]["progress"] = 95
        
        optimization_report = {
            'symbol': symbol,
            'strategy_type': strategy_type,
            'optimization_method': optimization_method,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'total_evaluations': len(optimization_results),
            'best_parameters': best_result['parameters'] if best_result else None,
            'best_metrics': best_result['metrics'] if best_result else None,
            'best_score': best_score,
            'validation_results': best_result.get('final_validation') if best_result else None,
            'top_results': sorted(optimization_results, key=lambda x: x['performance_score'], reverse=True)[:10] if optimization_results else []
        }
        
        # Store results
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["message"] = f"Strategy optimization completed for {strategy_type} on {symbol}"
        active_jobs[job_id]["progress"] = 100
        active_jobs[job_id]["completion_time"] = datetime.now().isoformat()
        active_jobs[job_id]["result"] = optimization_report
        
        logger.info(f"Completed internal optimization for {strategy_type} on {symbol}")
    
    except Exception as e:
        logger.error(f"Error in internal optimization: {str(e)}")
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["message"] = f"Error in optimization: {str(e)}"
