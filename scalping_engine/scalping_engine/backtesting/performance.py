import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger


class PerformanceAnalyzer:
    """
    Analysiert die Performance von Backtest-Ergebnissen.
    """
    
    def __init__(self, results_df: pd.DataFrame = None, trades: List[Dict[str, Any]] = None, metrics: Dict[str, Any] = None):
        """
        Initialisiert den Performance-Analyzer.
        
        Args:
            results_df: DataFrame mit Backtest-Ergebnissen
            trades: Liste der Trades
            metrics: Performance-Metriken
        """
        self.results = results_df.copy() if results_df is not None else None
        self.trades = trades or []
        self.metrics = metrics or {}
    
    @classmethod
    def from_file(cls, file_path: str) -> 'PerformanceAnalyzer':
        """
        Erstellt einen Performance-Analyzer aus einer gespeicherten Ergebnisdatei.
        
        Args:
            file_path: Pfad zur Ergebnisdatei
            
        Returns:
            PerformanceAnalyzer: Performance-Analyzer-Instanz
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"Datei {file_path} existiert nicht")
            return cls()
        
        # Je nach Dateityp laden
        if path.suffix.lower() == '.json':
            # JSON-Datei laden
            with open(path, 'r') as f:
                data = json.load(f)
            
            metrics = data.get('metrics', {})
            trades = data.get('trades', [])
            
            # Strategie-Typ und Parameter für Reporting extrahieren
            if 'config' in data:
                config = data['config']
                metrics['strategy_type'] = config.get('strategy_type', 'Unbekannt')
                metrics['strategy_params'] = config.get('strategy_params', {})
                metrics['symbol'] = config.get('symbol', 'Unbekannt')
                metrics['timeframe'] = config.get('timeframe', 'Unbekannt')
            
            # Prüfen, ob separate CSV-Datei existiert
            csv_path = path.with_suffix('.csv')
            if csv_path.exists():
                results_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            else:
                results_df = None
            
            return cls(results_df, trades, metrics)
        
        elif path.suffix.lower() == '.csv':
            # CSV-Datei laden
            results_df = pd.read_csv(path, index_col=0, parse_dates=True)
            
            # Prüfen, ob separate JSON-Datei existiert
            json_path = path.with_suffix('.json')
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                metrics = data.get('metrics', {})
                trades = data.get('trades', [])
                
                # Strategie-Typ und Parameter für Reporting extrahieren
                if 'config' in data:
                    config = data['config']
                    metrics['strategy_type'] = config.get('strategy_type', 'Unbekannt')
                    metrics['strategy_params'] = config.get('strategy_params', {})
                    metrics['symbol'] = config.get('symbol', 'Unbekannt')
                    metrics['timeframe'] = config.get('timeframe', 'Unbekannt')
            else:
                metrics = {}
                trades = []
            
            return cls(results_df, trades, metrics)
        
        else:
            logger.error(f"Unbekannter Dateityp: {path.suffix}")
            return cls()
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Berechnet Performance-Metriken aus den Backtest-Ergebnissen.
        
        Returns:
            Dict[str, Any]: Performance-Metriken
        """
        if self.results is None or self.results.empty:
            logger.error("Keine Ergebnisse zur Metrikberechnung vorhanden")
            return {}
        
        # Wenn bereits Metriken vorhanden sind, diese zurückgeben
        if self.metrics:
            return self.metrics
        
        # Equity-Kurve
        if 'equity' in self.results.columns:
            equity = self.results['equity']
            initial_capital = equity.iloc[0]
            final_equity = equity.iloc[-1]
            
            # Gesamtgewinn/-verlust
            total_profit_loss = final_equity - initial_capital
            total_profit_loss_pct = (total_profit_loss / initial_capital) * 100
            
            # Renditen
            returns = equity.pct_change().dropna()
            
            # Sharpe-Ratio (annualisiert)
            risk_free_rate = 0.02 / 252  # 2% pro Jahr, täglich
            excess_returns = returns - risk_free_rate
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
            
            # Sortino-Ratio (nur negative Renditen berücksichtigen)
            negative_returns = returns[returns < 0]
            sortino_ratio = (returns.mean() - risk_free_rate) / negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 and negative_returns.std() > 0 else 0
            
            # Drawdown
            if 'drawdown' in self.results.columns and 'drawdown_pct' in self.results.columns:
                max_drawdown = self.results['drawdown'].max()
                max_drawdown_pct = self.results['drawdown_pct'].max()
            else:
                # Drawdown berechnen, wenn nicht vorhanden
                peak = equity.cummax()
                drawdown = peak - equity
                drawdown_pct = (drawdown / peak) * 100
                max_drawdown = drawdown.max()
                max_drawdown_pct = drawdown_pct.max()
            
            # Recovery-Faktor
            recovery_factor = total_profit_loss / max_drawdown if max_drawdown > 0 else float('inf')
            
            # Calmar-Ratio (annualisierte Rendite / maximaler Drawdown)
            days = (self.results.index[-1] - self.results.index[0]).days
            annualized_return = ((final_equity / initial_capital) ** (365 / max(days, 1)) - 1) * 100
            calmar_ratio = annualized_return / max_drawdown_pct if max_drawdown_pct > 0 else float('inf')
            
            # Trades-Analyse
            num_trades = len(self.trades) if self.trades else 0
            
            if num_trades > 0:
                # In DataFrame konvertieren für einfachere Analyse
                trades_df = pd.DataFrame(self.trades)
                
                # Gewinnende/verlierende Trades
                winning_trades = trades_df[trades_df['profit_loss'] > 0]
                losing_trades = trades_df[trades_df['profit_loss'] <= 0]
                
                num_winning_trades = len(winning_trades)
                num_losing_trades = len(losing_trades)
                
                # Win-Rate
                win_rate = (num_winning_trades / num_trades) * 100
                
                # Durchschnittlicher Gewinn/Verlust
                avg_winning_trade = winning_trades['profit_loss'].mean() if len(winning_trades) > 0 else 0
                avg_losing_trade = losing_trades['profit_loss'].mean() if len(losing_trades) > 0 else 0
                
                # Gewinn-Verlust-Verhältnis
                avg_win_loss_ratio = abs(avg_winning_trade / avg_losing_trade) if avg_losing_trade != 0 else float('inf')
                
                # Profit-Faktor
                total_winning = winning_trades['profit_loss'].sum() if len(winning_trades) > 0 else 0
                total_losing = abs(losing_trades['profit_loss'].sum()) if len(losing_trades) > 0 else 0
                profit_factor = total_winning / total_losing if total_losing > 0 else float('inf')
                
                # Erwartungswert
                expectancy = (win_rate / 100 * avg_winning_trade) + ((1 - win_rate / 100) * avg_losing_trade)
                
                # Handelsrichtung
                if 'direction' in trades_df.columns:
                    long_trades = trades_df[trades_df['direction'] == 'long']
                    short_trades = trades_df[trades_df['direction'] == 'short']
                    
                    num_long_trades = len(long_trades)
                    num_short_trades = len(short_trades)
                    
                    long_win_rate = (len(long_trades[long_trades['profit_loss'] > 0]) / num_long_trades) * 100 if num_long_trades > 0 else 0
                    short_win_rate = (len(short_trades[short_trades['profit_loss'] > 0]) / num_short_trades) * 100 if num_short_trades > 0 else 0
                else:
                    num_long_trades = 0
                    num_short_trades = 0
                    long_win_rate = 0
                    short_win_rate = 0
                
                # Austrittsgründe
                if 'exit_reason' in trades_df.columns:
                    exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
                    stop_loss_exits = exit_reasons.get('stop_loss', 0)
                    take_profit_exits = exit_reasons.get('take_profit', 0)
                else:
                    stop_loss_exits = 0
                    take_profit_exits = 0
                
                # Trades pro Tag
                if days > 0:
                    trades_per_day = num_trades / days
                else:
                    trades_per_day = 0
                
                trades_metrics = {
                    'num_trades': num_trades,
                    'num_winning_trades': num_winning_trades,
                    'num_losing_trades': num_losing_trades,
                    'win_rate': win_rate,
                    'avg_winning_trade': avg_winning_trade,
                    'avg_losing_trade': avg_losing_trade,
                    'avg_win_loss_ratio': avg_win_loss_ratio,
                    'profit_factor': profit_factor,
                    'expectancy': expectancy,
                    'num_long_trades': num_long_trades,
                    'num_short_trades': num_short_trades,
                    'long_win_rate': long_win_rate,
                    'short_win_rate': short_win_rate,
                    'stop_loss_exits': stop_loss_exits,
                    'take_profit_exits': take_profit_exits,
                    'trades_per_day': trades_per_day
                }
            else:
                trades_metrics = {
                    'num_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'avg_winning_trade': 0,
                    'avg_losing_trade': 0,
                    'avg_win_loss_ratio': 0,
                    'expectancy': 0,
                    'num_long_trades': 0,
                    'num_short_trades': 0,
                    'long_win_rate': 0,
                    'short_win_rate': 0,
                    'stop_loss_exits': 0,
                    'take_profit_exits': 0,
                    'trades_per_day': 0
                }
            
            # Metriken zusammenstellen
            self.metrics = {
                'initial_capital': initial_capital,
                'final_equity': final_equity,
                'total_profit_loss': total_profit_loss,
                'total_profit_loss_pct': total_profit_loss_pct,
                'annualized_return': annualized_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown_pct,
                'recovery_factor': recovery_factor,
                'calmar_ratio': calmar_ratio,
                'days': days,
                **trades_metrics
            }
            
            return self.metrics
        
        else:
            logger.error("Keine Equity-Spalte in den Ergebnissen gefunden")
            return {}
    
    def plot_equity_curve(self, figsize: Tuple[int, int] = (12, 6), save_path: Optional[str] = None) -> None:
        """
        Erstellt einen Equity-Kurve-Plot.
        
        Args:
            figsize: Größe der Figur
            save_path: Pfad zum Speichern der Figur (optional)
        """
        if self.results is None or self.results.empty or 'equity' not in self.results.columns:
            logger.error("Keine Equity-Daten zum Plotten vorhanden")
            return
        
        # Metriken berechnen, falls noch nicht geschehen
        if not self.metrics:
            self.calculate_metrics()
        
        # Figur erstellen
        plt.figure(figsize=figsize)
        
        # Equity-Kurve
        plt.plot(self.results.index, self.results['equity'], label='Equity', color='blue')
        
        # Drawdown visualisieren
        if 'drawdown' in self.results.columns:
            plt.fill_between(
                self.results.index,
                self.results['equity'],
                self.results['equity'] + self.results['drawdown'],
                alpha=0.3,
                color='red',
                label='Drawdown'
            )
        
        # Handelssignale hinzufügen, falls vorhanden
        if 'signal' in self.results.columns:
            # Long-Signale
            long_signals = self.results[self.results['signal'] > 0].index
            if len(long_signals) > 0:
                plt.scatter(
                    long_signals,
                    self.results.loc[long_signals, 'equity'],
                    marker='^',
                    color='green',
                    label='Long Signal',
                    alpha=0.7
                )
            
            # Short-Signale
            short_signals = self.results[self.results['signal'] < 0].index
            if len(short_signals) > 0:
                plt.scatter(
                    short_signals,
                    self.results.loc[short_signals, 'equity'],
                    marker='v',
                    color='red',
                    label='Short Signal',
                    alpha=0.7
                )
        
        # Titel und Beschriftungen
        profit_loss_pct = self.metrics.get('total_profit_loss_pct', 0)
        max_drawdown_pct = self.metrics.get('max_drawdown_pct', 0)
        win_rate = self.metrics.get('win_rate', 0)
        strategy_type = self.metrics.get('strategy_type', 'Unbekannt')
        
        plt.title(f'Equity-Kurve {strategy_type}\nGewinn: {profit_loss_pct:.2f}%, Max DD: {max_drawdown_pct:.2f}%, Win Rate: {win_rate:.2f}%')
        plt.xlabel('Datum')
        plt.ylabel('Kapital')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Speichern, falls gewünscht
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_drawdown(self, figsize: Tuple[int, int] = (12, 6), save_path: Optional[str] = None) -> None:
        """
        Erstellt einen Drawdown-Plot.
        
        Args:
            figsize: Größe der Figur
            save_path: Pfad zum Speichern der Figur (optional)
        """
        if self.results is None or self.results.empty:
            logger.error("Keine Daten zum Plotten vorhanden")
            return
        
        # Drawdown berechnen, falls nicht vorhanden
        if 'drawdown_pct' not in self.results.columns and 'equity' in self.results.columns:
            equity = self.results['equity']
            peak = equity.cummax()
            self.results['drawdown'] = peak - equity
            self.results['drawdown_pct'] = (self.results['drawdown'] / peak) * 100
        
        if 'drawdown_pct' not in self.results.columns:
            logger.error("Keine Drawdown-Daten zum Plotten vorhanden")
            return
        
        # Figur erstellen
        plt.figure(figsize=figsize)
        
        # Drawdown-Kurve
        plt.fill_between(
            self.results.index,
            0,
            -self.results['drawdown_pct'],
            alpha=0.5,
            color='red'
        )
        plt.plot(self.results.index, -self.results['drawdown_pct'], color='darkred', label='Drawdown')
        
        # Titel und Beschriftungen
        max_drawdown_pct = self.results['drawdown_pct'].max()
        strategy_type = self.metrics.get('strategy_type', 'Unbekannt')
        
        plt.title(f'Drawdown {strategy_type}\nMaximaler Drawdown: {max_drawdown_pct:.2f}%')
        plt.xlabel('Datum')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        # Y-Achse invertieren für bessere Darstellung
        plt.gca().invert_yaxis()
        
        # Speichern, falls gewünscht
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_monthly_returns(self, figsize: Tuple[int, int] = (12, 6), save_path: Optional[str] = None) -> None:
        """
        Erstellt einen Plot mit monatlichen Renditen.
        
        Args:
            figsize: Größe der Figur
            save_path: Pfad zum Speichern der Figur (optional)
        """
        if self.results is None or self.results.empty or 'equity' not in self.results.columns:
            logger.error("Keine Equity-Daten zum Plotten vorhanden")
            return
        
        # Monatliche Renditen berechnen
        equity = self.results['equity']
        monthly_returns = equity.resample('M').last().pct_change().dropna() * 100
        
        if len(monthly_returns) < 2:
            logger.error("Nicht genügend Daten für monatliche Renditen")
            return
        
        # Figur erstellen
        plt.figure(figsize=figsize)
        
        # Farben je nach Rendite
        colors = ['green' if ret > 0 else 'red' for ret in monthly_returns]
        
        # Balkendiagramm
        plt.bar(monthly_returns.index, monthly_returns, color=colors, alpha=0.7)
        
        # Titel und Beschriftungen
        strategy_type = self.metrics.get('strategy_type', 'Unbekannt')
        
        plt.title(f'Monatliche Renditen {strategy_type}')
        plt.xlabel('Datum')
        plt.ylabel('Rendite (%)')
        plt.grid(True, alpha=0.3)
        
        # Nulllinie
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Durchschnittliche monatliche Rendite
        avg_return = monthly_returns.mean()
        plt.axhline(y=avg_return, color='blue', linestyle='--', alpha=0.7, label=f'Durchschnitt: {avg_return:.2f}%')
        
        plt.legend()
        
        # Speichern, falls gewünscht
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_trade_analysis(self, figsize: Tuple[int, int] = (15, 10), save_path: Optional[str] = None) -> None:
        """
        Erstellt einen Plot mit Trade-Analyse.
        
        Args:
            figsize: Größe der Figur
            save_path: Pfad zum Speichern der Figur (optional)
        """
        if not self.trades:
            logger.error("Keine Trades zum Analysieren vorhanden")
            return
        
        # Trades in DataFrame konvertieren
        trades_df = pd.DataFrame(self.trades)
        
        # Eintritts- und Austrittszeiten als Datetime konvertieren
        for col in ['entry_time', 'exit_time']:
            if col in trades_df.columns:
                if trades_df[col].dtype == 'object':
                    trades_df[col] = pd.to_datetime(trades_df[col])
        
        # Figur mit Subplots erstellen
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Gewinn/Verlust pro Trade
        ax1 = axes[0, 0]
        trades_df['profit_loss_color'] = ['green' if pl > 0 else 'red' for pl in trades_df['profit_loss']]
        ax1.bar(range(len(trades_df)), trades_df['profit_loss'], color=trades_df['profit_loss_color'], alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title('Gewinn/Verlust pro Trade')
        ax1.set_xlabel('Trade-Nummer')
        ax1.set_ylabel('Gewinn/Verlust')
        ax1.grid(True, alpha=0.3)
        
        # 2. Kumulativer Gewinn/Verlust
        ax2 = axes[0, 1]
        cumulative_pnl = trades_df['profit_loss'].cumsum()
        ax2.plot(range(len(trades_df)), cumulative_pnl, color='blue')
        ax2.set_title('Kumulativer Gewinn/Verlust')
        ax2.set_xlabel('Trade-Nummer')
        ax2.set_ylabel('Kumulativer G/V')
        ax2.grid(True, alpha=0.3)
        
        # 3. Verteilung der Gewinne/Verluste
        ax3 = axes[1, 0]
        sns.histplot(trades_df['profit_loss'], bins=20, kde=True, ax=ax3)
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('Verteilung der Gewinne/Verluste')
        ax3.set_xlabel('Gewinn/Verlust')
        ax3.set_ylabel('Häufigkeit')
        ax3.grid(True, alpha=0.3)
        
        # 4. Gewinn/Verlust nach Handelsrichtung
        ax4 = axes[1, 1]
        if 'direction' in trades_df.columns:
            direction_data = trades_df.groupby('direction')['profit_loss'].agg(['sum', 'mean', 'count'])
            directions = direction_data.index
            
            # Balkendiagramm für Gesamtgewinn/Verlust nach Richtung
            x = np.arange(len(directions))
            width = 0.25
            
            ax4.bar(x - width, direction_data['sum'], width, label='Gesamt')
            ax4.bar(x, direction_data['mean'], width, label='Durchschnitt')
            ax4.bar(x + width, direction_data['count'], width, label='Anzahl')
            
            ax4.set_xticks(x)
            ax4.set_xticklabels(directions)
            ax4.set_title('Gewinn/Verlust nach Handelsrichtung')
            ax4.set_xlabel('Richtung')
            ax4.set_ylabel('Wert')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "Keine Richtungsdaten vorhanden", ha='center', va='center')
            ax4.set_title('Gewinn/Verlust nach Handelsrichtung')
        
        # Haupttitel mit Strategie-Typ
        strategy_type = self.metrics.get('strategy_type', 'Unbekannt')
        fig.suptitle(f'Trade-Analyse für {strategy_type}', fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Platz für den Haupttitel
        
        # Speichern, falls gewünscht
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_exit_reasons(self, figsize: Tuple[int, int] = (10, 6), save_path: Optional[str] = None) -> None:
        """
        Erstellt einen Plot mit den Austrittsgrunden der Trades.
        
        Args:
            figsize: Größe der Figur
            save_path: Pfad zum Speichern der Figur (optional)
        """
        if not self.trades:
            logger.error("Keine Trades zum Analysieren vorhanden")
            return
        
        # Trades in DataFrame konvertieren
        trades_df = pd.DataFrame(self.trades)
        
        if 'exit_reason' not in trades_df.columns:
            logger.error("Keine Austrittsgründe in den Trades vorhanden")
            return
        
        # Austrittsgrunde zählen
        exit_reasons_count = trades_df['exit_reason'].value_counts()
        
        # Gewinn/Verlust nach Austrittsgrund
        exit_reason_pnl = trades_df.groupby('exit_reason')['profit_loss'].agg(['sum', 'mean', 'count'])
        
        # Figur mit Subplots erstellen
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Anzahl der Trades nach Austrittsgrund
        ax1.pie(exit_reasons_count, labels=exit_reasons_count.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')  # Gleiche Skalierung für Kreis
        ax1.set_title('Anteil der Austrittsgründe')
        
        # 2. Gewinn/Verlust nach Austrittsgrund
        exit_reason_pnl['sum'].plot(kind='bar', ax=ax2, color='blue', alpha=0.7, label='Gesamtgewinn/-verlust')
        ax2.set_title('Gewinn/Verlust nach Austrittsgrund')
        ax2.set_xlabel('Austrittsgrund')
        ax2.set_ylabel('Gewinn/Verlust')
        ax2.grid(True, alpha=0.3)
        
        # Haupttitel mit Strategie-Typ
        strategy_type = self.metrics.get('strategy_type', 'Unbekannt')
        fig.suptitle(f'Exit-Analyse für {strategy_type}', fontsize=14)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Platz für den Haupttitel
        
        # Speichern, falls gewünscht
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def generate_performance_report(self, output_dir: str = "reports") -> str:
        """
        Generiert einen umfassenden Performance-Bericht mit allen Plots und Metriken.
        
        Args:
            output_dir: Verzeichnis für die Ausgabedateien
            
        Returns:
            str: Pfad zum HTML-Bericht
        """
        # Verzeichnis erstellen, falls nicht vorhanden
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Eindeutigen Berichtnamen erstellen
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_type = self.metrics.get('strategy_type', 'unknown')
        report_name = f"performance_report_{strategy_type}_{timestamp}"
        report_dir = Path(output_dir) / report_name
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Metriken berechnen, falls noch nicht geschehen
        if not self.metrics:
            self.calculate_metrics()
        
        # Plots erstellen und speichern
        if self.results is not None and not self.results.empty:
            # Equity-Kurve
            equity_path = report_dir / "equity_curve.png"
            self.plot_equity_curve(save_path=str(equity_path))
            
            # Drawdown
            drawdown_path = report_dir / "drawdown.png"
            self.plot_drawdown(save_path=str(drawdown_path))
            
            # Monatliche Renditen
            monthly_returns_path = report_dir / "monthly_returns.png"
            self.plot_monthly_returns(save_path=str(monthly_returns_path))
        
# Trade-Analyse
        if self.trades:
            # Trade-Analyse
            trade_analysis_path = report_dir / "trade_analysis.png"
            self.plot_trade_analysis(save_path=str(trade_analysis_path))
            
            # Austrittsgründe
            exit_reasons_path = report_dir / "exit_reasons.png"
            self.plot_exit_reasons(save_path=str(exit_reasons_path))
        
        # Metriken in JSON-Datei speichern
        metrics_path = report_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        # HTML-Bericht erstellen
        html_path = report_dir / "report.html"
        
        # Strategie-spezifische Informationen
        strategy_type = self.metrics.get('strategy_type', 'Unbekannt')
        strategy_params = self.metrics.get('strategy_params', {})
        symbol = self.metrics.get('symbol', 'Unbekannt')
        timeframe = self.metrics.get('timeframe', 'Unbekannt')
        
        with open(html_path, 'w') as f:
            f.write(f'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Backtest Performance Report - {strategy_type}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .metric-container {{ display: flex; flex-wrap: wrap; }}
                    .metric-box {{ 
                        border: 1px solid #ddd; 
                        border-radius: 5px; 
                        padding: 10px; 
                        margin: 10px; 
                        width: 200px;
                        background-color: #f9f9f9;
                    }}
                    .strategy-box {{
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        padding: 15px;
                        margin: 15px 0;
                        background-color: #f5f5f5;
                    }}
                    .param-table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 10px;
                    }}
                    .param-table th, .param-table td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }}
                    .param-table th {{
                        background-color: #f2f2f2;
                    }}
                    .metric-box h3 {{ margin-top: 0; }}
                    .plot-container {{ margin: 20px 0; }}
                    .good {{ color: green; }}
                    .bad {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>Backtest Performance Report - {strategy_type}</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="strategy-box">
                    <h2>Strategie-Informationen</h2>
                    <p><strong>Strategie:</strong> {strategy_type}</p>
                    <p><strong>Symbol:</strong> {symbol}</p>
                    <p><strong>Zeitrahmen:</strong> {timeframe}</p>
                    
                    <h3>Strategie-Parameter</h3>
                    <table class="param-table">
                        <tr>
                            <th>Parameter</th>
                            <th>Wert</th>
                        </tr>
            ''')
            
            # Parameter in die Tabelle einfügen
            for key, value in strategy_params.items():
                f.write(f'''
                        <tr>
                            <td>{key}</td>
                            <td>{value}</td>
                        </tr>
                ''')
                
            f.write(f'''
                    </table>
                </div>
                
                <h2>Key Metrics</h2>
                <div class="metric-container">
                    <div class="metric-box">
                        <h3>Performance</h3>
                        <p>Initial Capital: <b>{self.metrics.get('initial_capital', 0):.2f}</b></p>
                        <p>Final Equity: <b>{self.metrics.get('final_equity', 0):.2f}</b></p>
                        <p>Total P/L: <b class="{'good' if self.metrics.get('total_profit_loss', 0) > 0 else 'bad'}">{self.metrics.get('total_profit_loss', 0):.2f}</b></p>
                        <p>Total P/L %: <b class="{'good' if self.metrics.get('total_profit_loss_pct', 0) > 0 else 'bad'}">{self.metrics.get('total_profit_loss_pct', 0):.2f}%</b></p>
                        <p>Annualized Return: <b class="{'good' if self.metrics.get('annualized_return', 0) > 0 else 'bad'}">{self.metrics.get('annualized_return', 0):.2f}%</b></p>
                    </div>
                    
                    <div class="metric-box">
                        <h3>Risk Metrics</h3>
                        <p>Max Drawdown: <b class="bad">{self.metrics.get('max_drawdown', 0):.2f}</b></p>
                        <p>Max Drawdown %: <b class="bad">{self.metrics.get('max_drawdown_pct', 0):.2f}%</b></p>
                        <p>Sharpe Ratio: <b>{self.metrics.get('sharpe_ratio', 0):.2f}</b></p>
                        <p>Sortino Ratio: <b>{self.metrics.get('sortino_ratio', 0):.2f}</b></p>
                        <p>Calmar Ratio: <b>{self.metrics.get('calmar_ratio', 0):.2f}</b></p>
                    </div>
                    
                    <div class="metric-box">
                        <h3>Trade Statistics</h3>
                        <p>Total Trades: <b>{self.metrics.get('num_trades', 0)}</b></p>
                        <p>Win Rate: <b class="{'good' if self.metrics.get('win_rate', 0) > 50 else 'bad'}">{self.metrics.get('win_rate', 0):.2f}%</b></p>
                        <p>Profit Factor: <b>{self.metrics.get('profit_factor', 0):.2f}</b></p>
                        <p>Recovery Factor: <b>{self.metrics.get('recovery_factor', 0):.2f}</b></p>
                        <p>Expectancy: <b>{self.metrics.get('expectancy', 0):.2f}</b></p>
                    </div>
                    
                    <div class="metric-box">
                        <h3>Trade Details</h3>
                        <p>Winning Trades: <b class="good">{self.metrics.get('num_winning_trades', 0)}</b></p>
                        <p>Losing Trades: <b class="bad">{self.metrics.get('num_losing_trades', 0)}</b></p>
                        <p>Avg. Winning Trade: <b class="good">{self.metrics.get('avg_winning_trade', 0):.2f}</b></p>
                        <p>Avg. Losing Trade: <b class="bad">{self.metrics.get('avg_losing_trade', 0):.2f}</b></p>
                        <p>Win/Loss Ratio: <b>{self.metrics.get('avg_win_loss_ratio', 0):.2f}</b></p>
                    </div>
                    
                    <div class="metric-box">
                        <h3>Direction & Exits</h3>
                        <p>Long Trades: <b>{self.metrics.get('num_long_trades', 0)}</b> 
                           (Win: <b>{self.metrics.get('long_win_rate', 0):.2f}%</b>)</p>
                        <p>Short Trades: <b>{self.metrics.get('num_short_trades', 0)}</b> 
                           (Win: <b>{self.metrics.get('short_win_rate', 0):.2f}%</b>)</p>
                        <p>Stop Loss Exits: <b>{self.metrics.get('stop_loss_exits', 0)}</b></p>
                        <p>Take Profit Exits: <b>{self.metrics.get('take_profit_exits', 0)}</b></p>
                        <p>Trades per Day: <b>{self.metrics.get('trades_per_day', 0):.2f}</b></p>
                    </div>
                </div>
                
                <h2>Equity & Drawdown</h2>
                <div class="plot-container">
                    <img src="equity_curve.png" alt="Equity Curve" style="max-width: 100%;">
                </div>
                
                <div class="plot-container">
                    <img src="drawdown.png" alt="Drawdown" style="max-width: 100%;">
                </div>
                
                <h2>Returns</h2>
                <div class="plot-container">
                    <img src="monthly_returns.png" alt="Monthly Returns" style="max-width: 100%;">
                </div>
                
                <h2>Trade Analysis</h2>
                <div class="plot-container">
                    <img src="trade_analysis.png" alt="Trade Analysis" style="max-width: 100%;">
                </div>
                
                <div class="plot-container">
                    <img src="exit_reasons.png" alt="Exit Reasons" style="max-width: 100%;">
                </div>
                
                <h2>All Metrics</h2>
                <pre>{json.dumps(self.metrics, indent=2, default=str)}</pre>
            </body>
            </html>
            ''')
        
        logger.info(f"Performance-Bericht erstellt: {html_path}")
        
        return str(html_path)
    
    def print_summary(self) -> None:
        """
        Gibt eine Zusammenfassung der Performance aus.
        """
        # Metriken berechnen, falls noch nicht geschehen
        if not self.metrics:
            self.calculate_metrics()
        
        strategy_type = self.metrics.get('strategy_type', 'Unbekannt')
        symbol = self.metrics.get('symbol', 'Unbekannt')
        timeframe = self.metrics.get('timeframe', 'Unbekannt')
        
        print(f"\n===== Performance-Zusammenfassung für {strategy_type} ({symbol}/{timeframe}) =====")
        print(f"Anfangskapital: {self.metrics.get('initial_capital', 0):.2f}")
        print(f"Endkapital: {self.metrics.get('final_equity', 0):.2f}")
        print(f"Gewinn/Verlust: {self.metrics.get('total_profit_loss', 0):.2f} ({self.metrics.get('total_profit_loss_pct', 0):.2f}%)")
        print(f"Annualisierte Rendite: {self.metrics.get('annualized_return', 0):.2f}%")
        print(f"Maximaler Drawdown: {self.metrics.get('max_drawdown', 0):.2f} ({self.metrics.get('max_drawdown_pct', 0):.2f}%)")
        print(f"Anzahl Trades: {self.metrics.get('num_trades', 0)}")
        print(f"Win-Rate: {self.metrics.get('win_rate', 0):.2f}%")
        print(f"Profit-Faktor: {self.metrics.get('profit_factor', 0):.2f}")
        print(f"Sharpe-Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Sortino-Ratio: {self.metrics.get('sortino_ratio', 0):.2f}")
        print(f"Calmar-Ratio: {self.metrics.get('calmar_ratio', 0):.2f}")
        print(f"Recovery-Faktor: {self.metrics.get('recovery_factor', 0):.2f}")
        print("=======================================\n")


# Hilfsfunktionen für externe Aufrufe
async def analyze_backtest(backtest_id: int, save_report: bool = True) -> Dict[str, Any]:
    """
    Analysiert einen Backtest und erstellt optional einen Bericht.
    
    Args:
        backtest_id: ID des Backtests in der Datenbank
        save_report: Ob ein Bericht erstellt werden soll
        
    Returns:
        Dict[str, Any]: Metriken und Pfad zum Bericht
    """
    from scalping_engine.backtesting.engine import backtest_engine
    
    # Backtest-Daten laden
    backtest_data = await backtest_engine.load_backtest(backtest_id)
    
    if not backtest_data:
        return {"error": f"Backtest mit ID {backtest_id} nicht gefunden"}
    
    # Ergebnisdaten abrufen
    results_data = backtest_data.get('results', {})
    metrics = backtest_data.get('metrics', {})
    trades = backtest_data.get('trades', [])
    
    # Strategie-Informationen für Reporting hinzufügen
    metrics['strategy_type'] = backtest_data.get('strategy', {}).get('name', 'Unbekannt').split('(')[0].strip()
    metrics['strategy_params'] = backtest_data.get('strategy', {}).get('parameters', {})
    metrics['symbol'] = backtest_data.get('symbol', 'Unbekannt')
    metrics['timeframe'] = backtest_data.get('timeframe', 'Unbekannt')
    
    # Performance-Analyzer erstellen
    analyzer = PerformanceAnalyzer(metrics=metrics, trades=trades)
    
    # Bericht erstellen, falls gewünscht
    report_path = None
    if save_report:
        report_path = analyzer.generate_performance_report(output_dir=f"reports/backtest_{backtest_id}")
    
    # Zusammenfassung ausgeben
    analyzer.print_summary()
    
    return {
        "metrics": metrics,
        "report_path": report_path
    }


# Hilfsfunktion zum Vergleichen mehrerer Strategien
def compare_strategies(backtest_results: List[Dict[str, Any]], output_dir: str = "reports") -> str:
    """
    Vergleicht mehrere Backtests verschiedener Strategien und erstellt einen Vergleichsbericht.
    
    Args:
        backtest_results: Liste von Backtest-Ergebnissen
        output_dir: Verzeichnis für die Ausgabedateien
        
    Returns:
        str: Pfad zum Vergleichsbericht
    """
    if not backtest_results:
        logger.error("Keine Backtest-Ergebnisse zum Vergleichen")
        return ""
    
    # Verzeichnis erstellen, falls nicht vorhanden
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Eindeutigen Berichtnamen erstellen
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"strategy_comparison_{timestamp}"
    report_dir = Path(output_dir) / report_name
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Vergleichstabelle erstellen
    comparison_data = []
    for result in backtest_results:
        metrics = result.get('metrics', {})
        strategy_type = metrics.get('strategy_type', 'Unbekannt')
        symbol = metrics.get('symbol', 'Unbekannt')
        timeframe = metrics.get('timeframe', 'Unbekannt')
        
        comparison_data.append({
            'strategy': strategy_type,
            'symbol': symbol,
            'timeframe': timeframe,
            'total_profit_loss_pct': metrics.get('total_profit_loss_pct', 0),
            'max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
            'win_rate': metrics.get('win_rate', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'num_trades': metrics.get('num_trades', 0)
        })
    
    # DataFrame erstellen und nach Gesamtgewinn sortieren
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('total_profit_loss_pct', ascending=False)
    
    # CSV-Datei speichern
    csv_path = report_dir / "strategy_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    
    # Vergleichsdiagramme erstellen
    # 1. Gesamtgewinn/-verlust
    plt.figure(figsize=(12, 6))
    plt.bar(comparison_df['strategy'], comparison_df['total_profit_loss_pct'], color='blue', alpha=0.7)
    plt.title('Gesamtgewinn/-verlust (%)')
    plt.xlabel('Strategie')
    plt.ylabel('Gewinn/Verlust (%)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_dir / "profit_comparison.png", dpi=100)
    
    # 2. Drawdown-Vergleich
    plt.figure(figsize=(12, 6))
    plt.bar(comparison_df['strategy'], comparison_df['max_drawdown_pct'], color='red', alpha=0.7)
    plt.title('Maximaler Drawdown (%)')
    plt.xlabel('Strategie')
    plt.ylabel('Drawdown (%)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_dir / "drawdown_comparison.png", dpi=100)
    
    # 3. Win-Rate-Vergleich
    plt.figure(figsize=(12, 6))
    plt.bar(comparison_df['strategy'], comparison_df['win_rate'], color='green', alpha=0.7)
    plt.title('Win-Rate (%)')
    plt.xlabel('Strategie')
    plt.ylabel('Win-Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_dir / "winrate_comparison.png", dpi=100)
    
    # 4. Radar-Chart für mehrere Metriken
    metrics_to_plot = ['total_profit_loss_pct', 'win_rate', 'profit_factor', 'sharpe_ratio']
    metrics_labels = ['Gewinn (%)', 'Win-Rate (%)', 'Profit-Faktor', 'Sharpe-Ratio']
    
    # Daten normalisieren für den Radar-Chart
    normalized_data = comparison_df[metrics_to_plot].copy()
    for col in normalized_data.columns:
        min_val = normalized_data[col].min()
        max_val = normalized_data[col].max()
        if max_val > min_val:
            normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        else:
            normalized_data[col] = 0
    
    # Radar-Chart erstellen
    plt.figure(figsize=(10, 8))
    
    # Anzahl der Metriken
    N = len(metrics_to_plot)
    
    # Winkel für jede Metrik
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Geschlossenes Polygon
    
    # Radar-Chart-Plot
    ax = plt.subplot(111, polar=True)
    
    # Für jede Strategie plotten
    for i, strategy in enumerate(comparison_df['strategy']):
        values = normalized_data.iloc[i].tolist()
        values += values[:1]  # Geschlossenes Polygon
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=strategy)
        ax.fill(angles, values, alpha=0.1)
    
    # Labels und Legende
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels)
    ax.set_yticklabels([])
    ax.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Strategie-Vergleich')
    plt.tight_layout()
    plt.savefig(report_dir / "strategy_radar_chart.png", dpi=100)
    
    # HTML-Bericht erstellen
    html_path = report_dir / "comparison_report.html"
    
    with open(html_path, 'w') as f:
        f.write(f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Strategie-Vergleich</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 20px 0;
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 8px; 
                    text-align: right;
                }}
                th {{ 
                    background-color: #f2f2f2; 
                    text-align: center;
                }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                tr:hover {{ background-color: #f2f2f2; }}
                .plot-container {{ margin: 20px 0; }}
                .best {{ background-color: #d4edda; }}
                .worst {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <h1>Strategie-Vergleich</h1>
            <p>Erstellt am: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Vergleichstabelle</h2>
            <table>
                <tr>
                    <th>Strategie</th>
                    <th>Symbol</th>
                    <th>Zeitrahmen</th>
                    <th>Gewinn/Verlust (%)</th>
                    <th>Max. Drawdown (%)</th>
                    <th>Win-Rate (%)</th>
                    <th>Profit-Faktor</th>
                    <th>Sharpe-Ratio</th>
                    <th>Anzahl Trades</th>
                </tr>
        ''')
        
        # Zeilen der Tabelle
        for i, row in comparison_df.iterrows():
            best_class = ' class="best"' if i == 0 else ''
            worst_class = ' class="worst"' if i == len(comparison_df) - 1 else ''
            
            f.write(f'''
                <tr{best_class if i == 0 else worst_class if i == len(comparison_df) - 1 else ""}>
                    <td>{row['strategy']}</td>
                    <td>{row['symbol']}</td>
                    <td>{row['timeframe']}</td>
                    <td>{row['total_profit_loss_pct']:.2f}%</td>
                    <td>{row['max_drawdown_pct']:.2f}%</td>
                    <td>{row['win_rate']:.2f}%</td>
                    <td>{row['profit_factor']:.2f}</td>
                    <td>{row['sharpe_ratio']:.2f}</td>
                    <td>{row['num_trades']}</td>
                </tr>
            ''')
            
        f.write('''
            </table>
            
            <h2>Gewinn-/Verlust-Vergleich</h2>
            <div class="plot-container">
                <img src="profit_comparison.png" alt="Gewinn-/Verlust-Vergleich" style="max-width: 100%;">
            </div>
            
            <h2>Drawdown-Vergleich</h2>
            <div class="plot-container">
                <img src="drawdown_comparison.png" alt="Drawdown-Vergleich" style="max-width: 100%;">
            </div>
            
            <h2>Win-Rate-Vergleich</h2>
            <div class="plot-container">
                <img src="winrate_comparison.png" alt="Win-Rate-Vergleich" style="max-width: 100%;">
            </div>
            
            <h2>Strategie-Radar-Chart</h2>
            <div class="plot-container">
                <img src="strategy_radar_chart.png" alt="Strategie-Radar-Chart" style="max-width: 100%;">
            </div>
        </body>
        </html>
        ''')
    
    logger.info(f"Strategie-Vergleichsbericht erstellt: {html_path}")
    
    return str(html_path)
