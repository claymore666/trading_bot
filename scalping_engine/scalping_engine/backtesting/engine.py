import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Type
from datetime import datetime, timedelta
from loguru import logger
import concurrent.futures
import asyncio
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

# Aktualisierte Importe für die BacktestingEngine
from scalping_engine.data_manager.fetcher import fetch_market_data
from scalping_engine.strategy_engine.strategy_base import StrategyBase, load_strategies
from scalping_engine.models.backtest import Backtest
from scalping_engine.utils.db import get_async_db

@dataclass
class BacktestConfig:
    """
    Konfiguration für einen Backtest.
    """
    strategy_id: Optional[int] = None
    strategy_type: str = "BollingerBandsStrategy"  # oder "MovingAverageCrossStrategy"
    strategy_params: Dict[str, Any] = None
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = 10000.0
    commission: float = 0.001  # 0.1% pro Trade
    risk_per_trade: float = 1.0
    risk_reward_ratio: float = 2.0
    atr_multiplier: float = 2.0


class BacktestingEngine:
    """
    Engine zum Durchführen, Speichern und Analysieren von Backtests.
    """
    
    def __init__(self):
        """
        Initialisiert die Backtesting-Engine.
        """
        # Stelle sicher, dass die Strategien geladen sind
        self._load_strategies()
        
        # Verzeichnisse für Ergebnisse erstellen
        os.makedirs("data/backtest_results", exist_ok=True)
    
    def _load_strategies(self):
        """
        Lädt alle verfügbaren Strategien, falls sie noch nicht geladen sind.
        """
        # Strategien laden, falls nicht bereits geschehen
        if not hasattr(StrategyBase, '_strategy_registry') or not StrategyBase._strategy_registry:
            try:
                # Versuche, die Lade-Funktion zu importieren und auszuführen
                from scalping_engine.strategy_engine.strategy_base import load_strategies
                load_strategies()
                logger.info(f"Strategien für Backtesting geladen. Verfügbar: {', '.join(StrategyBase._strategy_registry.keys())}")
            except Exception as e:
                logger.error(f"Fehler beim Laden der Strategien: {str(e)}")
    
    def _create_strategy_instance(self, config: BacktestConfig) -> StrategyBase:
        """
        Erstellt eine Strategie-Instanz basierend auf der Konfiguration.
        
        Args:
            config: Backtesting-Konfiguration mit Strategie-Typ und -Parametern
            
        Returns:
            StrategyBase: Instanz der angegebenen Strategie
        """
        try:
            # Strategie aus dem Registry holen
            strategy = StrategyBase.create_strategy(
                config.strategy_type,
                name=f"{config.strategy_type} ({config.symbol}/{config.timeframe})",
                description=f"Backtesting-Strategie für {config.symbol}/{config.timeframe}",
                parameters=config.strategy_params or {},
                risk_per_trade=config.risk_per_trade
            )
            return strategy
        except Exception as e:
            available_strategies = ', '.join(StrategyBase._strategy_registry.keys()) if hasattr(StrategyBase, '_strategy_registry') else "Keine"
            raise ValueError(f"Unbekannter Strategie-Typ: {config.strategy_type}. Verfügbare Strategien: {available_strategies}. Fehler: {str(e)}")
    
    async def fetch_data_for_backtest(self, config: BacktestConfig) -> pd.DataFrame:
        """
        Ruft die Daten für den Backtest ab.
        
        Args:
            config: Backtesting-Konfiguration mit Symbol, Timeframe, Start- und Enddatum
            
        Returns:
            pd.DataFrame: OHLCV-Daten für den Backtest
        """
        # Standardwerte für Start- und Enddatum
        if not config.end_date:
            config.end_date = datetime.now()
        if not config.start_date:
            config.start_date = config.end_date - timedelta(days=30)
        
        # Daten abrufen
        data = await fetch_market_data(
            symbol=config.symbol,
            timeframe=config.timeframe,
            start_date=config.start_date,
            end_date=config.end_date
        )
        
        if data.empty:
            logger.error(f"Keine Daten für {config.symbol}/{config.timeframe} abgerufen")
            return pd.DataFrame()
        
        return data
    
    async def run_backtest(self, config: BacktestConfig) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Führt einen Backtest durch.
        
        Args:
            config: Backtesting-Konfiguration
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]: Backtest-Ergebnisse, Metriken und Trades
        """
        # Strategie-Instanz erstellen
        strategy = self._create_strategy_instance(config)
        
        # Daten abrufen
        data = await self.fetch_data_for_backtest(config)
        
        if data.empty:
            logger.error(f"Keine Daten für Backtest verfügbar: {config.symbol}/{config.timeframe}")
            return pd.DataFrame(), {}, []
        
        # Strategie initialisieren
        strategy.initialize(data)
        
        # Backtest durchführen
        results, metrics = strategy.backtest(
            initial_capital=config.initial_capital,
            commission=config.commission,
            risk_reward_ratio=config.risk_reward_ratio,
            atr_multiplier=config.atr_multiplier
        )
        
        # Trades extrahieren
        trades = []
        if 'trades' in metrics:
            trades = metrics['trades']
        
        return results, metrics, trades
    
    async def save_backtest(
        self,
        config: BacktestConfig,
        results: pd.DataFrame,
        metrics: Dict[str, Any],
        db: Optional[AsyncSession] = None
    ) -> int:
        """
        Speichert die Backtest-Ergebnisse in der Datenbank.
        
        Args:
            config: Backtesting-Konfiguration
            results: Backtest-Ergebnisse
            metrics: Performance-Metriken
            db: Datenbanksession (optional)
            
        Returns:
            int: ID des gespeicherten Backtests
        """
        # Datenbank-Session abrufen, falls nicht übergeben
        close_db = False
        if db is None:
            db_gen = get_async_db()
            db = await anext(db_gen)
            close_db = True
        
        try:
            # Strategie in der Datenbank suchen oder erstellen
            from scalping_engine.models.strategy import Strategy
            
            strategy_query = select(Strategy).where(Strategy.name == f"{config.strategy_type} ({config.symbol}/{config.timeframe})")
            strategy_result = await db.execute(strategy_query)
            strategy = strategy_result.scalar_one_or_none()
            
            # Neue Strategie erstellen, falls keine gefunden
            if not strategy:
                strategy = Strategy(
                    name=f"{config.strategy_type} ({config.symbol}/{config.timeframe})",
                    description=f"Backtesting-Strategie für {config.symbol}/{config.timeframe}",
                    parameters=config.strategy_params or {},
                    is_active=True,
                    is_optimized=False
                )
                db.add(strategy)
                await db.flush()
            
            # Backtest-Eintrag erstellen
            backtest = Backtest(
                strategy_id=strategy.id,
                symbol=config.symbol,
                timeframe=config.timeframe,
                start_date=config.start_date,
                end_date=config.end_date,
                initial_capital=config.initial_capital,
                status="completed",
                completed_at=datetime.now(),
                results={
                    "config": asdict(config),
                    "summary": metrics
                },
                metrics=metrics,
                trades=[]  # Trades werden später hinzugefügt
            )
            
            db.add(backtest)
            await db.commit()
            
            logger.info(f"Backtest gespeichert mit ID {backtest.id}")
            
            return backtest.id
        
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Backtests: {str(e)}")
            await db.rollback()
            return 0
        
        finally:
            if close_db:
                await db.close()
    
    async def load_backtest(self, backtest_id: int, db: Optional[AsyncSession] = None) -> Dict[str, Any]:
        """
        Lädt einen Backtest aus der Datenbank.
        
        Args:
            backtest_id: ID des Backtests
            db: Datenbanksession (optional)
            
        Returns:
            Dict[str, Any]: Backtest-Daten
        """
        # Datenbank-Session abrufen, falls nicht übergeben
        close_db = False
        if db is None:
            db_gen = get_async_db()
            db = await anext(db_gen)
            close_db = True
        
        try:
            # Backtest aus der Datenbank abrufen
            query = select(Backtest).where(Backtest.id == backtest_id)
            result = await db.execute(query)
            backtest = result.scalar_one_or_none()
            
            if not backtest:
                logger.error(f"Kein Backtest mit ID {backtest_id} gefunden")
                return {}
            
            # Strategie abrufen
            from scalping_engine.models.strategy import Strategy
            strategy_query = select(Strategy).where(Strategy.id == backtest.strategy_id)
            strategy_result = await db.execute(strategy_query)
            strategy = strategy_result.scalar_one_or_none()
            
            # Backtest-Daten zusammenstellen
            backtest_data = {
                "id": backtest.id,
                "strategy": {
                    "id": strategy.id if strategy else None,
                    "name": strategy.name if strategy else "Unbekannte Strategie",
                    "parameters": strategy.parameters if strategy else {}
                },
                "symbol": backtest.symbol,
                "timeframe": backtest.timeframe,
                "start_date": backtest.start_date,
                "end_date": backtest.end_date,
                "initial_capital": backtest.initial_capital,
                "status": backtest.status,
                "completed_at": backtest.completed_at,
                "results": backtest.results,
                "metrics": backtest.metrics,
                "trades": backtest.trades or []
            }
            
            return backtest_data
        
        except Exception as e:
            logger.error(f"Fehler beim Laden des Backtests: {str(e)}")
            return {}
        
        finally:
            if close_db:
                await db.close()
    
    async def run_multiple_backtests(
        self,
        configs: List[BacktestConfig],
        parallel: bool = True,
        max_workers: int = 4
    ) -> List[Tuple[BacktestConfig, Dict[str, Any]]]:
        """
        Führt mehrere Backtests parallel oder sequentiell durch.
        
        Args:
            configs: Liste von Backtesting-Konfigurationen
            parallel: Ob die Backtests parallel ausgeführt werden sollen
            max_workers: Maximale Anzahl paralleler Arbeiter
            
        Returns:
            List[Tuple[BacktestConfig, Dict[str, Any]]]: Liste von Konfigurationen und Metriken
        """
        results = []
        
        if parallel and len(configs) > 1:
            # Parallelisierung nur auf CPU-gebundene Aufgaben anwenden
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Future-Objekte erstellen
                futures = []
                for config in configs:
                    future = executor.submit(self._run_backtest_sync, config)
                    futures.append((config, future))
                
                # Ergebnisse sammeln
                for config, future in futures:
                    _, metrics, _ = future.result()
                    results.append((config, metrics))
        else:
            # Sequentielle Ausführung
            for config in configs:
                _, metrics, _ = await self.run_backtest(config)
                results.append((config, metrics))
        
        return results
    
    def _run_backtest_sync(self, config: BacktestConfig) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Synchrone Version von run_backtest für Parallelisierung.
        
        Args:
            config: Backtesting-Konfiguration
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]: Ergebnisse, Metriken und Trades
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(self.run_backtest(config))
        finally:
            loop.close()
    
    def analyze_results(self, results: List[Tuple[BacktestConfig, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analysiert die Ergebnisse mehrerer Backtests.
        
        Args:
            results: Liste von Konfigurationen und Metriken
            
        Returns:
            Dict[str, Any]: Analyseergebnisse
        """
        # Grundlegende Statistiken
        num_backtests = len(results)
        if num_backtests == 0:
            return {"error": "Keine Backtest-Ergebnisse zur Analyse"}
        
        # Durchschnittliche Metriken
        avg_profit = sum(metrics.get('total_profit_loss_pct', 0) for _, metrics in results) / num_backtests
        avg_win_rate = sum(metrics.get('win_rate', 0) for _, metrics in results) / num_backtests
        avg_profit_factor = sum(metrics.get('profit_factor', 0) for _, metrics in results) / num_backtests
        avg_max_drawdown = sum(metrics.get('max_drawdown_pct', 0) for _, metrics in results) / num_backtests
        
        # Beste und schlechteste Performance
        results_sorted_by_profit = sorted(results, key=lambda x: x[1].get('total_profit_loss_pct', 0), reverse=True)
        best_result = results_sorted_by_profit[0] if results_sorted_by_profit else None
        worst_result = results_sorted_by_profit[-1] if len(results_sorted_by_profit) > 1 else None
        
        # Verteilung der Metriken
        profit_distribution = [metrics.get('total_profit_loss_pct', 0) for _, metrics in results]
        win_rate_distribution = [metrics.get('win_rate', 0) for _, metrics in results]
        
        # Analyse nach Strategie-Typ
        strategy_types = set(config.strategy_type for config, _ in results)
        strategy_analysis = {}
        
        for strategy_type in strategy_types:
            strategy_results = [(config, metrics) for config, metrics in results if config.strategy_type == strategy_type]
            avg_strategy_profit = sum(metrics.get('total_profit_loss_pct', 0) for _, metrics in strategy_results) / len(strategy_results)
            
            strategy_analysis[strategy_type] = {
                "count": len(strategy_results),
                "avg_profit_pct": avg_strategy_profit
            }
        
        # Analyseausgabe
        analysis = {
            "num_backtests": num_backtests,
            "avg_metrics": {
                "profit_pct": avg_profit,
                "win_rate": avg_win_rate,
                "profit_factor": avg_profit_factor,
                "max_drawdown_pct": avg_max_drawdown
            },
            "best_result": {
                "config": asdict(best_result[0]) if best_result else None,
                "metrics": best_result[1] if best_result else None
            },
            "worst_result": {
                "config": asdict(worst_result[0]) if worst_result else None,
                "metrics": worst_result[1] if worst_result else None
            },
            "distributions": {
                "profit_pct": {
                    "min": min(profit_distribution) if profit_distribution else None,
                    "max": max(profit_distribution) if profit_distribution else None,
                    "median": sorted(profit_distribution)[len(profit_distribution) // 2] if profit_distribution else None
                },
                "win_rate": {
                    "min": min(win_rate_distribution) if win_rate_distribution else None,
                    "max": max(win_rate_distribution) if win_rate_distribution else None,
                    "median": sorted(win_rate_distribution)[len(win_rate_distribution) // 2] if win_rate_distribution else None
                }
            },
            "strategy_analysis": strategy_analysis
        }
        
        return analysis
    
    def save_results_to_file(
        self, 
        results: pd.DataFrame, 
        metrics: Dict[str, Any], 
        config: BacktestConfig,
        trades: List[Dict[str, Any]] = None
    ) -> str:
        """
        Speichert die Backtest-Ergebnisse in einer Datei.
        
        Args:
            results: Backtest-Ergebnisse
            metrics: Performance-Metriken
            config: Backtesting-Konfiguration
            trades: Liste der Trades (optional)
            
        Returns:
            str: Pfad der gespeicherten Datei
        """
        # Eindeutige ID für die Ergebnisdatei
        result_id = str(uuid.uuid4())[:8]
        
        # Verzeichnis erstellen, falls nicht vorhanden
        result_dir = Path("data/backtest_results")
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Dateinamen erstellen
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config.strategy_type}_{config.symbol.replace('/', '_')}_{config.timeframe}_{timestamp}_{result_id}"
        
        # Metriken und Konfiguration speichern
        result_data = {
            "config": asdict(config),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Trades hinzufügen, falls vorhanden
        if trades:
            result_data["trades"] = trades
        
        # JSON-Datei speichern
        json_path = result_dir / f"{filename}.json"
        with open(json_path, "w") as f:
            json.dump(result_data, f, indent=2, default=str)
        
        # CSV-Datei speichern (nur die wichtigsten Spalten)
        csv_path = result_dir / f"{filename}.csv"
        csv_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'signal', 'position', 'capital', 'equity', 'drawdown', 'drawdown_pct'
        ]
        
        # Nur Spalten speichern, die tatsächlich im DataFrame vorhanden sind
        available_columns = [col for col in csv_columns if col in results.columns]
        results[available_columns].to_csv(csv_path)
        
        logger.info(f"Backtest-Ergebnisse gespeichert: {json_path}")
        
        return str(json_path)


# Hilfsfunktionen für externe Aufrufe
backtest_engine = BacktestingEngine()

async def run_backtest(
    strategy_id: Optional[int] = None,
    strategy_type: str = "BollingerBandsStrategy",
    strategy_params: Dict[str, Any] = None,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    risk_per_trade: float = 1.0,
    save_results: bool = True,
    db: Optional[AsyncSession] = None
) -> Dict[str, Any]:
    """
    Hilfsfunktion zum Durchführen eines Backtests.
    
    Args:
        strategy_id: ID der Strategie in der Datenbank (optional)
        strategy_type: Typ der Strategie
        strategy_params: Parameter der Strategie
        symbol: Handelssymbol
        timeframe: Zeitrahmen
        start_date: Startdatum
        end_date: Enddatum
        initial_capital: Anfangskapital
        commission: Kommission pro Trade
        risk_per_trade: Risiko pro Trade in Prozent des Kapitals
        save_results: Ob die Ergebnisse gespeichert werden sollen
        db: Datenbanksession (optional)
        
    Returns:
        Dict[str, Any]: Backtest-Ergebnisse
    """
    # Backtesting-Konfiguration erstellen
    config = BacktestConfig(
        strategy_id=strategy_id,
        strategy_type=strategy_type,
        strategy_params=strategy_params,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        commission=commission,
        risk_per_trade=risk_per_trade
    )
    
    # Backtest durchführen
    results, metrics, trades = await backtest_engine.run_backtest(config)
    
    if results.empty:
        return {"error": "Backtest konnte nicht durchgeführt werden"}
    
    # Ergebnisse speichern, falls gewünscht
    if save_results:
        # In Datei speichern
        file_path = backtest_engine.save_results_to_file(results, metrics, config, trades)
        
        # In Datenbank speichern
        if db:
            backtest_id = await backtest_engine.save_backtest(config, results, metrics, db)
            metrics['backtest_id'] = backtest_id
        
        metrics['file_path'] = file_path
    
    # Wichtigste Metriken für die Rückgabe auswählen
    result_summary = {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy_type": strategy_type,
        "period": f"{start_date} to {end_date}",
        "initial_capital": initial_capital,
        "final_equity": metrics.get('final_equity', 0),
        "total_profit_loss": metrics.get('total_profit_loss', 0),
        "total_profit_loss_pct": metrics.get('total_profit_loss_pct', 0),
        "max_drawdown": metrics.get('max_drawdown', 0),
        "max_drawdown_pct": metrics.get('max_drawdown_pct', 0),
        "num_trades": metrics.get('num_trades', 0),
        "win_rate": metrics.get('win_rate', 0),
        "profit_factor": metrics.get('profit_factor', 0),
        "sharpe_ratio": metrics.get('sharpe_ratio', 0),
        "recovery_factor": metrics.get('recovery_factor', 0),
        "metrics": metrics
    }
    
    return result_summary
