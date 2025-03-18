import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Type
from loguru import logger
from datetime import datetime
from abc import ABC, abstractmethod

# Annahme: Diese Funktion existiert in einem anderen Modul
from scalping_engine.data_manager.processor import process_market_data


class StrategyBase(ABC):
    """
    Basisklasse für alle Handelsstrategien.
    Definiert die gemeinsame Schnittstelle und Funktionalität.
    """
    
    # Registry für alle verfügbaren Strategien
    _strategy_registry = {}
    
    @classmethod
    def register(cls, strategy_class):
        """
        Dekorator zum Registrieren einer Strategie.
        
        Args:
            strategy_class: Die zu registrierende Strategieklasse
            
        Returns:
            Die registrierte Strategieklasse
        """
        cls._strategy_registry[strategy_class.__name__] = strategy_class
        logger.debug(f"Strategie registriert: {strategy_class.__name__}")
        return strategy_class
    
    @classmethod
    def get_strategy_class(cls, strategy_name: str) -> Type['StrategyBase']:
        """
        Gibt die Strategieklasse mit dem angegebenen Namen zurück.
        
        Args:
            strategy_name: Name der Strategieklasse
            
        Returns:
            Type['StrategyBase']: Die gesuchte Strategieklasse
            
        Raises:
            ValueError: Wenn keine Strategie mit dem angegebenen Namen gefunden wurde
        """
        if strategy_name not in cls._strategy_registry:
            available_strategies = ', '.join(cls._strategy_registry.keys())
            raise ValueError(f"Keine Strategie mit Namen '{strategy_name}' gefunden. "
                           f"Verfügbare Strategien: {available_strategies}")
        
        return cls._strategy_registry[strategy_name]
    
    @classmethod
    def create_strategy(cls, strategy_name: str, **kwargs) -> 'StrategyBase':
        """
        Erzeugt eine Instanz der Strategie mit dem angegebenen Namen.
        
        Args:
            strategy_name: Name der Strategieklasse
            **kwargs: Parameter für den Strategiekonstruktor
            
        Returns:
            StrategyBase: Eine Instanz der angegebenen Strategie
        """
        strategy_class = cls.get_strategy_class(strategy_name)
        return strategy_class(**kwargs)
    
    @classmethod
    def list_available_strategies(cls) -> List[Dict[str, Any]]:
        """
        Listet alle verfügbaren Strategien mit ihren Beschreibungen auf.
        
        Returns:
            List[Dict[str, Any]]: Liste von Strategieinformationen
        """
        strategies_info = []
        
        for name, strategy_class in cls._strategy_registry.items():
            strategies_info.append({
                "name": name,
                "description": strategy_class.__doc__ or "Keine Beschreibung verfügbar"
            })
        
        return strategies_info
    
    def __init__(
        self,
        name: str,
        description: str = "",
        parameters: Dict[str, Any] = None,
        risk_per_trade: float = 1.0
    ):
        """
        Initialisiert die Basisstrategie.
        
        Args:
            name: Name der Strategie
            description: Beschreibung der Strategie
            parameters: Parameter der Strategie
            risk_per_trade: Risiko pro Trade in Prozent des Kapitals
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.risk_per_trade = risk_per_trade
        
        # Interne Zustandsvariablen
        self.is_initialized = False
        self._data = None
    
    def initialize(self, data: pd.DataFrame) -> None:
        """
        Initialisiert die Strategie mit den Marktdaten.
        
        Args:
            data: OHLCV-Daten als DataFrame
        """
        if data.empty:
            logger.error("Leere Daten für Strategie-Initialisierung")
            self.is_initialized = False
            return
        
        # Marktdaten vorverarbeiten und Features hinzufügen
        self._data = process_market_data(
            df=data,
            clean=True,
            add_features=True,
            add_volume=True,
            detect_market_gaps=True
        )
        
        self.is_initialized = True
        logger.info(f"Strategie {self.name} initialisiert mit {len(self._data)} Datenpunkten")
    
    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """
        Generiert Handelssignale basierend auf den initialisierten Daten.
        Muss von den abgeleiteten Klassen implementiert werden.
        
        Returns:
            pd.DataFrame: DataFrame mit Signalen (1 für Long, -1 für Short, 0 für neutral)
        """
        pass
    
    def calculate_position_size(self, price: float, stop_loss: float, capital: float) -> float:
        """
        Berechnet die Positionsgröße basierend auf dem Risiko pro Trade.
        
        Args:
            price: Aktueller Preis
            stop_loss: Stop-Loss-Preis
            capital: Verfügbares Kapital
            
        Returns:
            float: Positionsgröße in Einheiten
        """
        # Minimaler Stop-Loss für numerische Stabilität
        min_price_diff = price * 0.0001  # 0.01% des Preises
        
        # Absolute Differenz zwischen Preis und Stop-Loss berechnen
        price_diff = abs(price - stop_loss)
        price_diff = max(price_diff, min_price_diff)  # Nulldivision verhindern
        
        # Risikobetrag berechnen
        risk_amount = capital * (self.risk_per_trade / 100)
        
        # Positionsgröße berechnen
        position_size = risk_amount / price_diff
        
        return position_size
    
    def calculate_stop_loss(
        self, 
        price: float, 
        direction: int, 
        atr_multiplier: float = 2.0, 
        custom_atr: Optional[float] = None
    ) -> float:
        """
        Berechnet den Stop-Loss-Preis basierend auf ATR.
        
        Args:
            price: Aktueller Preis
            direction: Handelsrichtung (1 für Long, -1 für Short)
            atr_multiplier: Multiplikator für den ATR-Wert
            custom_atr: Optionaler benutzerdefinierter ATR-Wert
            
        Returns:
            float: Stop-Loss-Preis
        """
        if not self.is_initialized or self._data is None or self._data.empty:
            logger.error("Strategie nicht initialisiert oder keine Daten vorhanden")
            return price * 0.95 if direction > 0 else price * 1.05
        
        # ATR verwenden, falls in den Daten vorhanden
        if custom_atr is not None:
            atr = custom_atr
        elif 'true_range' in self._data.columns:
            atr = self._data['true_range'].iloc[-1]
        else:
            # Fallback: Nimm 2% des Preises als "ATR"
            atr = price * 0.02
            logger.warning("Kein ATR in Daten gefunden, verwende 2% des Preises als Fallback")
        
        # Stop-Loss-Preis berechnen
        stop_loss = price - (direction * atr_multiplier * atr)
        
        return stop_loss
    
    def calculate_take_profit(
        self, 
        price: float, 
        direction: int, 
        risk_reward_ratio: float = 2.0,
        stop_loss: Optional[float] = None
    ) -> float:
        """
        Berechnet den Take-Profit-Preis basierend auf dem Risiko-Rendite-Verhältnis.
        
        Args:
            price: Aktueller Preis
            direction: Handelsrichtung (1 für Long, -1 für Short)
            risk_reward_ratio: Gewünschtes Risiko-Rendite-Verhältnis
            stop_loss: Optionaler benutzerdefinierter Stop-Loss-Preis
            
        Returns:
            float: Take-Profit-Preis
        """
        if stop_loss is None:
            stop_loss = self.calculate_stop_loss(price, direction)
        
        # Risiko berechnen (Abstand vom Einstieg zum Stop-Loss)
        risk = abs(price - stop_loss)
        
        # Take-Profit als Vielfaches des Risikos
        take_profit = price + (direction * risk * risk_reward_ratio)
        
        return take_profit
    
    def backtest(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,  # 0.1% pro Trade
        risk_reward_ratio: float = 2.0,
        atr_multiplier: float = 2.0
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Führt einen Backtest der Strategie durch.
        
        Args:
            initial_capital: Anfangskapital
            commission: Kommission pro Trade
            risk_reward_ratio: Gewünschtes Risiko-Rendite-Verhältnis
            atr_multiplier: Multiplikator für ATR bei der Stop-Loss-Berechnung
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: DataFrame mit Backtest-Ergebnissen und Metriken
        """
        if not self.is_initialized or self._data is None or self._data.empty:
            logger.error("Strategie nicht initialisiert oder keine Daten vorhanden")
            return pd.DataFrame(), {}
        
        # Signale generieren
        signals = self.generate_signals()
        
        if signals.empty:
            logger.error("Keine Handelssignale generiert")
            return pd.DataFrame(), {}
        
        # DataFrame für Backtest-Ergebnisse vorbereiten
        results = signals.copy()
        
        # Spalten für Backtest-Ergebnisse hinzufügen
        results['position'] = 0.0
        results['capital'] = initial_capital
        results['equity'] = initial_capital
        results['returns'] = 0.0
        results['drawdown'] = 0.0
        results['drawdown_pct'] = 0.0
        
        # Aktuelle Position und Kapital
        position = 0.0
        capital = initial_capital
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        peak_equity = initial_capital
        entry_time = None
        
        # Trades-Liste für detaillierte Analyse
        trades = []
        
        # Backtest durchführen
        for i, (idx, row) in enumerate(results.iterrows()):
            # Aktueller Preis
            price = row['close']
            
            # ATR für Stop-Loss-Berechnung
            atr = row['true_range'] if 'true_range' in row else price * 0.02
            
            # Signal für diesen Zeitpunkt
            signal = row['signal']
            
            # Wenn eine aktive Position besteht, prüfen wir, ob Stop-Loss oder Take-Profit erreicht wurde
            if position != 0:
                # Für Long-Positionen
                if position > 0:
                    # Stop-Loss erreicht?
                    if row['low'] <= stop_loss:
                        # Trade schließen zum Stop-Loss-Preis
                        trade_result = (stop_loss / entry_price - 1) * 100
                        trade_profit = position * entry_price * (stop_loss / entry_price - 1)
                        capital += position * stop_loss * (1 - commission) - position * entry_price * (1 + commission)
                        
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': idx,
                            'direction': 'long',
                            'entry_price': entry_price,
                            'exit_price': stop_loss,
                            'position_size': position,
                            'profit_loss': trade_profit,
                            'profit_loss_pct': trade_result,
                            'exit_reason': 'stop_loss'
                        })
                        
                        position = 0
                        logger.debug(f"Long-Position geschlossen bei Stop-Loss {stop_loss:.2f}, P/L: {trade_result:.2f}%")
                    
                    # Take-Profit erreicht?
                    elif row['high'] >= take_profit:
                        # Trade schließen zum Take-Profit-Preis
                        trade_result = (take_profit / entry_price - 1) * 100
                        trade_profit = position * entry_price * (take_profit / entry_price - 1)
                        capital += position * take_profit * (1 - commission) - position * entry_price * (1 + commission)
                        
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': idx,
                            'direction': 'long',
                            'entry_price': entry_price,
                            'exit_price': take_profit,
                            'position_size': position,
                            'profit_loss': trade_profit,
                            'profit_loss_pct': trade_result,
                            'exit_reason': 'take_profit'
                        })
                        
                        position = 0
                        logger.debug(f"Long-Position geschlossen bei Take-Profit {take_profit:.2f}, P/L: {trade_result:.2f}%")
                
                # Für Short-Positionen
                elif position < 0:
                    # Stop-Loss erreicht?
                    if row['high'] >= stop_loss:
                        # Trade schließen zum Stop-Loss-Preis
                        trade_result = (entry_price / stop_loss - 1) * 100
                        trade_profit = -position * entry_price * (entry_price / stop_loss - 1)
                        capital += -position * entry_price * (1 - commission) - (-position) * stop_loss * (1 + commission)
                        
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': idx,
                            'direction': 'short',
                            'entry_price': entry_price,
                            'exit_price': stop_loss,
                            'position_size': -position,
                            'profit_loss': trade_profit,
                            'profit_loss_pct': trade_result,
                            'exit_reason': 'stop_loss'
                        })
                        
                        position = 0
                        logger.debug(f"Short-Position geschlossen bei Stop-Loss {stop_loss:.2f}, P/L: {trade_result:.2f}%")
                    
                    # Take-Profit erreicht?
                    elif row['low'] <= take_profit:
                        # Trade schließen zum Take-Profit-Preis
                        trade_result = (entry_price / take_profit - 1) * 100
                        trade_profit = -position * entry_price * (entry_price / take_profit - 1)
                        capital += -position * entry_price * (1 - commission) - (-position) * take_profit * (1 + commission)
                        
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': idx,
                            'direction': 'short',
                            'entry_price': entry_price,
                            'exit_price': take_profit,
                            'position_size': -position,
                            'profit_loss': trade_profit,
                            'profit_loss_pct': trade_result,
                            'exit_reason': 'take_profit'
                        })
                        
                        position = 0
                        logger.debug(f"Short-Position geschlossen bei Take-Profit {take_profit:.2f}, P/L: {trade_result:.2f}%")
            
            # Wenn keine Position besteht und ein Signal vorliegt, eröffnen wir eine neue Position
            if position == 0 and signal != 0:
                # Long-Position eröffnen
                if signal > 0:
                    # Stop-Loss berechnen
                    stop_loss = self.calculate_stop_loss(price, 1, atr_multiplier, atr)
                    
                    # Take-Profit berechnen
                    take_profit = self.calculate_take_profit(price, 1, risk_reward_ratio, stop_loss)
                    
                    # Positionsgröße berechnen
                    position_size = self.calculate_position_size(price, stop_loss, capital)
                    
                    # Position eröffnen
                    position = position_size
                    entry_price = price
                    entry_time = idx
                    
                    logger.debug(f"Long-Position eröffnet bei {price:.2f}, Stop-Loss: {stop_loss:.2f}, Take-Profit: {take_profit:.2f}")
                
                # Short-Position eröffnen
                elif signal < 0:
                    # Stop-Loss berechnen
                    stop_loss = self.calculate_stop_loss(price, -1, atr_multiplier, atr)
                    
                    # Take-Profit berechnen
                    take_profit = self.calculate_take_profit(price, -1, risk_reward_ratio, stop_loss)
                    
                    # Positionsgröße berechnen
                    position_size = self.calculate_position_size(price, stop_loss, capital)
                    
                    # Position eröffnen
                    position = -position_size
                    entry_price = price
                    entry_time = idx
                    
                    logger.debug(f"Short-Position eröffnet bei {price:.2f}, Stop-Loss: {stop_loss:.2f}, Take-Profit: {take_profit:.2f}")
            
            # Aktuelle Position und Equity aktualisieren
            results.loc[idx, 'position'] = position
            
            # Wenn eine Position besteht, aktualisieren wir die Equity basierend auf dem aktuellen Preis
            if position != 0:
                if position > 0:  # Long-Position
                    results.loc[idx, 'equity'] = capital + position * (price * (1 - commission) - entry_price * (1 + commission))
                else:  # Short-Position
                    results.loc[idx, 'equity'] = capital + (-position) * (entry_price * (1 - commission) - price * (1 + commission))
            else:
                results.loc[idx, 'equity'] = capital
            
            # Kapital aktualisieren
            results.loc[idx, 'capital'] = capital
            
            # Maximum-Equity aktualisieren
            peak_equity = max(peak_equity, results.loc[idx, 'equity'])
            
            # Drawdown berechnen
            results.loc[idx, 'drawdown'] = peak_equity - results.loc[idx, 'equity']
            results.loc[idx, 'drawdown_pct'] = (results.loc[idx, 'drawdown'] / peak_equity) * 100
        
        # Letzte offene Position schließen
        if position != 0:
            last_price = results.iloc[-1]['close']
            
            if position > 0:  # Long-Position
                trade_result = (last_price / entry_price - 1) * 100
                trade_profit = position * entry_price * (last_price / entry_price - 1)
                capital += position * last_price * (1 - commission) - position * entry_price * (1 + commission)
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': results.index[-1],
                    'direction': 'long',
                    'entry_price': entry_price,
                    'exit_price': last_price,
                    'position_size': position,
                    'profit_loss': trade_profit,
                    'profit_loss_pct': trade_result,
                    'exit_reason': 'end_of_data'
                })
            else:  # Short-Position
                trade_result = (entry_price / last_price - 1) * 100
                trade_profit = -position * entry_price * (entry_price / last_price - 1)
                capital += -position * entry_price * (1 - commission) - (-position) * last_price * (1 + commission)
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': results.index[-1],
                    'direction': 'short',
                    'entry_price': entry_price,
                    'exit_price': last_price,
                    'position_size': -position,
                    'profit_loss': trade_profit,
                    'profit_loss_pct': trade_result,
                    'exit_reason': 'end_of_data'
                })
            
            # Letzte Equity und Kapital aktualisieren
            results.loc[results.index[-1], 'equity'] = capital
            results.loc[results.index[-1], 'capital'] = capital
        
        # Handels-DataFrame erstellen
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Metriken berechnen
        metrics = self._calculate_performance_metrics(results, trades_df, initial_capital)
        
        return results, metrics
    
    def _calculate_performance_metrics(
        self,
        results: pd.DataFrame,
        trades_df: pd.DataFrame,
        initial_capital: float
    ) -> Dict[str, Any]:
        """
        Berechnet Performance-Metriken für den Backtest.
        
        Args:
            results: DataFrame mit Backtest-Ergebnissen
            trades_df: DataFrame mit Trades
            initial_capital: Anfangskapital
            
        Returns:
            Dict[str, Any]: Performance-Metriken
        """
        if results.empty:
            return {}
        
        # Finale Equity
        final_equity = results.iloc[-1]['equity']
        
        # Gesamtgewinn/-verlust
        total_profit_loss = final_equity - initial_capital
        total_profit_loss_pct = (total_profit_loss / initial_capital) * 100
        
        # Maximaler Drawdown
        max_drawdown = results['drawdown'].max()
        max_drawdown_pct = results['drawdown_pct'].max()
        
        # Anzahl der Trades
        num_trades = len(trades_df) if not trades_df.empty else 0
        
        # Gewinnende und verlierende Trades
        if not trades_df.empty:
            winning_trades = trades_df[trades_df['profit_loss'] > 0]
            losing_trades = trades_df[trades_df['profit_loss'] <= 0]
            
            num_winning_trades = len(winning_trades)
            num_losing_trades = len(losing_trades)
            
            # Win-Rate
            win_rate = (num_winning_trades / num_trades) * 100 if num_trades > 0 else 0
            
            # Durchschnittlicher Gewinn und Verlust
            avg_winning_trade = winning_trades['profit_loss'].mean() if not winning_trades.empty else 0
            avg_losing_trade = losing_trades['profit_loss'].mean() if not losing_trades.empty else 0
            
            # Profit-Faktor
            total_winning = winning_trades['profit_loss'].sum() if not winning_trades.empty else 0
            total_losing = abs(losing_trades['profit_loss'].sum()) if not losing_trades.empty else 0
            profit_factor = total_winning / total_losing if total_losing > 0 else float('inf')
        else:
            num_winning_trades = 0
            num_losing_trades = 0
            win_rate = 0
            avg_winning_trade = 0
            avg_losing_trade = 0
            profit_factor = 0
        
        # Sharpe-Ratio (vereinfacht)
        if len(results) > 1:
            daily_returns = results['equity'].pct_change().dropna()
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 0 and daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Recovery-Faktor
        recovery_factor = total_profit_loss / max_drawdown if max_drawdown > 0 else float('inf')
        
        # Sortino-Ratio (nur negative Renditen berücksichtigen)
        if len(results) > 1:
            daily_returns = results['equity'].pct_change().dropna()
            negative_returns = daily_returns[daily_returns < 0]
            sortino_ratio = (daily_returns.mean() / negative_returns.std()) * np.sqrt(252) if len(negative_returns) > 0 and negative_returns.std() > 0 else 0
        else:
            sortino_ratio = 0
        
        # Metriken zusammenstellen
        metrics = {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_profit_loss': total_profit_loss,
            'total_profit_loss_pct': total_profit_loss_pct,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'num_trades': num_trades,
            'num_winning_trades': num_winning_trades,
            'num_losing_trades': num_losing_trades,
            'win_rate': win_rate,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'recovery_factor': recovery_factor,
            'trades': trades_df.to_dict('records') if not trades_df.empty else []
        }
        
        return metrics


# Funktion zum dynamischen Laden und Registrieren aller Strategien
def load_strategies():
    """
    Lädt alle verfügbaren Strategien aus dem strategy_engine Modul.
    
    Returns:
        List[str]: Liste der geladenen Strategienamen
    """
    import importlib
    import pkgutil
    import inspect
    import sys
    
    loaded_strategies = []
    
    try:
        # Das strategy_engine Paket importieren
        import scalping_engine.strategy_engine as strategy_engine_pkg
        
        # Alle Module im Paket durchlaufen
        for _, module_name, _ in pkgutil.iter_modules(strategy_engine_pkg.__path__):
            # Nur Strategiemodule berücksichtigen (Module, die auf "strategy" enden)
            if not module_name.endswith('strategy') and not module_name.endswith('detector') and not module_name.endswith('filter'):
                continue
            
            # Modul importieren
            module = importlib.import_module(f"scalping_engine.strategy_engine.{module_name}")
            
            # Alle Klassen im Modul durchlaufen
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, StrategyBase) and obj != StrategyBase:
                    # Klasse registrieren
                    StrategyBase.register(obj)
                    loaded_strategies.append(name)
                    logger.debug(f"Strategie {name} aus Modul {module_name} geladen")
        
        logger.info(f"Insgesamt {len(loaded_strategies)} Strategien geladen")
        
    except Exception as e:
        logger.error(f"Fehler beim Laden der Strategien: {str(e)}")
    
    return loaded_strategies
