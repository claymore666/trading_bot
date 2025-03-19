#!/usr/bin/env python3
"""
Hauptprogramm für das Modulare Hochperformante Scalping-System.

Führt den Benutzer durch den gesamten Prozess:
1. Auswahl des Assets
2. Analyse des Assets und Auswahl einer passenden Strategie
3. Initialisierung der Strategie und Durchführung des Backtests
4. Ausgabe der Backtest-Ergebnisse
"""

import asyncio
import argparse
from datetime import datetime, timedelta
from loguru import logger

from scalping_engine.data_manager.fetcher import fetch_market_data
from scalping_engine.data_manager.processor import process_market_data
from scalping_engine.data_manager.storage import DataStorage
from scalping_engine.data_manager.market_analyzer import analyze_market_and_recommend_strategies
from scalping_engine.backtesting.engine import BacktestingEngine, BacktestConfig
from scalping_engine.strategy_engine.strategy_base import StrategyBase, load_strategies
from scalping_engine.utils.db import get_async_db


async def main():
    """Hauptfunktion für das Scalping-System."""
    # Befehlszeilenargumente einrichten
    parser = argparse.ArgumentParser(description="Modulares Hochperformantes Scalping-System")
    parser.add_argument("--symbol", type=str, help="Symbol des zu handelnden Assets (z.B. 'BTC/USDT')")
    parser.add_argument("--days", type=int, default=180, help="Anzahl der Tage für historische Daten")
    
    args = parser.parse_args()
    
    # Verfügbare Strategien laden
    load_strategies()
    strategy_list = StrategyBase.list_available_strategies()
    logger.info(f"Verfügbare Strategien: {len(strategy_list)}")
    
    # Asset auswählen (vom Benutzer oder interaktiv)
    symbol = args.symbol
    if not symbol:
        # Vorhandene Symbole anzeigen
        db_gen = get_async_db()
        db = await anext(db_gen)
        available_symbols = await DataStorage.get_available_symbols(db)
        await db.close()
        
        if available_symbols:
            print("\nVerfügbare Assets:")
            for i, sym in enumerate(available_symbols):
                print(f"{i+1}. {sym}")
            
            choice = input("Wählen Sie ein Asset (Nummer oder Symbol): ")
            
            if choice.isdigit() and 1 <= int(choice) <= len(available_symbols):
                symbol = available_symbols[int(choice) - 1]
            elif choice in available_symbols:
                symbol = choice
            else:
                print("Ungültige Auswahl. Bitte geben Sie ein neues Symbol ein.")
                symbol = input("Asset-Symbol (z.B. 'BTC/USDT'): ")
        else:
            symbol = input("Asset-Symbol (z.B. 'BTC/USDT'): ")
    
    logger.info(f"Ausgewähltes Asset: {symbol}")
    
    # Daten laden
    print(f"\nLade Daten für {symbol}...")
    timeframes = ["5m", "15m", "1h", "4h", "1d"]
    data = {}
    
    start_date = datetime.now() - timedelta(days=args.days)
    end_date = datetime.now()
    
    for timeframe in timeframes:
        print(f"Lade {timeframe} Daten...")
        df = await fetch_market_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            print(f"Keine Daten für {symbol}/{timeframe} gefunden.")
            continue
        
        # Daten verarbeiten
        processed_df = process_market_data(df=df, clean=True)
        data[timeframe] = processed_df
        print(f"✓ {len(processed_df)} Datenpunkte für {timeframe} geladen.")
    
    if not data:
        print(f"Keine Daten für {symbol} gefunden. Bitte ein anderes Asset wählen.")
        return
    
    # Asset analysieren
    print(f"\nAnalysiere {symbol}...")
    
    # Vollständige Marktanalyse mit dem neuen Modul durchführen
    market_analysis = analyze_market_and_recommend_strategies(symbol, data)
    
    # Analyseergebnisse ausgeben
    print(f"\n=== Asset-Analyse ===")
    print(f"Symbol: {symbol}")
    print(f"Markttyp: {market_analysis['summary']['market_type']}")
    print(f"Volatilität: {market_analysis['summary']['volatility']}")
    print(f"Trendstärke: {market_analysis['summary']['trend_strength']}")
    print(f"Mean Reversion: {market_analysis['summary']['mean_reversion']}")
    
    # Strategieempfehlungen
    recommended_strategies = market_analysis['recommended_strategies']
    
    print("\n=== Empfohlene Strategien ===")
    for i, strategy in enumerate(recommended_strategies):
        print(f"{i+1}. {strategy['strategy']} - {strategy['reason']}")
    
    # Strategie auswählen
    if recommended_strategies:
        choice = input("\nWählen Sie eine Strategie (Nummer, Standard: 1): ") or "1"
        
        if choice.isdigit() and 1 <= int(choice) <= len(recommended_strategies):
            selected_strategy = recommended_strategies[int(choice) - 1]
            strategy_type = selected_strategy['strategy']
            params = selected_strategy.get('suggested_params', {})
        else:
            strategy_type = recommended_strategies[0]['strategy']
            params = recommended_strategies[0].get('suggested_params', {})
    else:
        print("Keine empfohlenen Strategien verfügbar.")
        print("\nVerfügbare Strategien:")
        for i, st in enumerate(strategy_list):
            print(f"{i+1}. {st['name']}")
        
        choice = input("Wählen Sie eine Strategie (Nummer): ")
        if choice.isdigit() and 1 <= int(choice) <= len(strategy_list):
            strategy_type = strategy_list[int(choice) - 1]["name"]
            params = {}
        else:
            print("Ungültige Auswahl.")
            return
    
    print(f"\nAusgewählte Strategie: {strategy_type}")
    
    # Primären Zeitrahmen für die Strategie bestimmen
    if strategy_type in ["BollingerBandsStrategy", "MeanReversionStrategy"]:
        preferred_tf = ["15m", "5m", "1h"]
    elif strategy_type in ["MovingAverageCrossStrategy", "ParabolicSARStrategy"]:
        preferred_tf = ["1h", "4h", "15m"]
    elif strategy_type == "BreakoutStrategy":
        preferred_tf = ["5m", "15m", "1h"]
    else:
        preferred_tf = ["1h", "15m", "5m"]
    
    # Das erste verfügbare bevorzugte Zeitintervall wählen
    strategy_tf = next((tf for tf in preferred_tf if tf in data), list(data.keys())[0])
    
    # Strategie initialisieren
    print(f"Initialisiere Strategie mit Daten für Zeitrahmen {strategy_tf}...")
    
    try:
        strategy = StrategyBase.create_strategy(
            strategy_type,
            name=f"{strategy_type} ({symbol})",
            description=f"Automatisch generierte Strategie für {symbol}",
            parameters=params
        )
        
        # Strategie mit Daten initialisieren
        strategy.initialize(data[strategy_tf])
        
        print("Strategie erfolgreich initialisiert.")
        
        # Backtest durchführen
        print("\nFühre Backtest durch...")
        results, metrics = strategy.backtest(
            initial_capital=10000.0,
            commission=0.001,  # 0.1% pro Trade
            risk_per_trade=1.0,
            risk_reward_ratio=2.0,
            atr_multiplier=2.0
        )
        
        # Ergebnisse anzeigen
        print("\n=== Backtest-Ergebnisse ===")
        print(f"Anfangskapital: {metrics.get('initial_capital', 0):.2f}")
        print(f"Endkapital: {metrics.get('final_equity', 0):.2f}")
        print(f"Gewinn/Verlust: {metrics.get('total_profit_loss', 0):.2f} ({metrics.get('total_profit_loss_pct', 0):.2f}%)")
        print(f"Maximaler Drawdown: {metrics.get('max_drawdown', 0):.2f} ({metrics.get('max_drawdown_pct', 0):.2f}%)")
        print(f"Trades: {metrics.get('num_trades', 0)}")
        print(f"Gewonnene Trades: {metrics.get('num_winning_trades', 0)}")
        print(f"Verlorene Trades: {metrics.get('num_losing_trades', 0)}")
        print(f"Win-Rate: {metrics.get('win_rate', 0):.2f}%")
        print(f"Profit-Faktor: {metrics.get('profit_factor', 0):.2f}")
        print(f"Sharpe-Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Sortino-Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        
        # Beurteilung der Strategie
        print("\n=== Strategie-Beurteilung ===")
        
        if metrics.get('total_profit_loss_pct', 0) > 0:
            print("✓ Die Strategie ist profitabel.")
        else:
            print("✗ Die Strategie ist nicht profitabel.")
        
        if metrics.get('win_rate', 0) >= 50:
            print(f"✓ Die Win-Rate von {metrics.get('win_rate', 0):.2f}% ist gut.")
        else:
            print(f"✗ Die Win-Rate von {metrics.get('win_rate', 0):.2f}% ist niedrig.")
        
        if metrics.get('profit_factor', 0) >= 1.5:
            print(f"✓ Der Profit-Faktor von {metrics.get('profit_factor', 0):.2f} ist gut.")
        else:
            print(f"✗ Der Profit-Faktor von {metrics.get('profit_factor', 0):.2f} ist niedrig.")
        
        if metrics.get('sharpe_ratio', 0) >= 1.0:
            print(f"✓ Die Sharpe-Ratio von {metrics.get('sharpe_ratio', 0):.2f} ist gut.")
        else:
            print(f"✗ Die Sharpe-Ratio von {metrics.get('sharpe_ratio', 0):.2f} ist niedrig.")
        
        if metrics.get('max_drawdown_pct', 0) <= 15:
            print(f"✓ Der maximale Drawdown von {metrics.get('max_drawdown_pct', 0):.2f}% ist akzeptabel.")
        else:
            print(f"✗ Der maximale Drawdown von {metrics.get('max_drawdown_pct', 0):.2f}% ist hoch.")
        
        if metrics.get('num_trades', 0) >= 30:
            print(f"✓ Die Anzahl der Trades ({metrics.get('num_trades', 0)}) ist ausreichend für eine statistische Beurteilung.")
        else:
            print(f"✗ Die Anzahl der Trades ({metrics.get('num_trades', 0)}) ist zu gering für eine zuverlässige statistische Beurteilung.")
        
        # Gesamtbeurteilung
        score = 0
        if metrics.get('total_profit_loss_pct', 0) > 0: score += 1
        if metrics.get('win_rate', 0) >= 50: score += 1
        if metrics.get('profit_factor', 0) >= 1.5: score += 1
        if metrics.get('sharpe_ratio', 0) >= 1.0: score += 1
        if metrics.get('max_drawdown_pct', 0) <= 15: score += 1
        if metrics.get('num_trades', 0) >= 30: score += 1
        
        print(f"\nGesamtbewertung: {score}/6 Punkte")
        
        if score >= 5:
            print("Die Strategie scheint gut zu funktionieren und kann für Optimierung verwendet werden.")
        elif score >= 3:
            print("Die Strategie zeigt Potential, aber sollte optimiert werden.")
        else:
            print("Die Strategie zeigt keine überzeugenden Ergebnisse. Eine andere Strategie könnte besser geeignet sein.")
        
    except Exception as e:
        print(f"Fehler beim Backtest: {str(e)}")
        return

if __name__ == "__main__":
    asyncio.run(main())
