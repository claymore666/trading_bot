#!/usr/bin/env python3
"""
Strategy Optimizer for the High-Performance Modular Scalping System.

This script provides various optimization methods for trading strategies:
1. Grid Search - exhaustive search over specified parameter ranges
2. Genetic Algorithm - evolutionary approach for parameter optimization
3. ML-based Optimization - uses machine learning to guide the parameter search
"""

import asyncio
import argparse
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import os
import sys
from typing import Dict, Any, List, Optional, Tuple

from scalping_engine.data_manager.fetcher import fetch_market_data
from scalping_engine.data_manager.processor import process_market_data
from scalping_engine.backtesting.engine import BacktestConfig, backtest_engine
from scalping_engine.strategy_engine.strategy_base import StrategyBase, load_strategies
from scalping_engine.utils.logger import setup_logging

# Try to import sklearn for ML optimization
try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class StrategyOptimizer:
    """
    Strategy optimizer that implements multiple optimization methods.
    """
    
    def __init__(self, strategy_type: str, symbol: str, timeframe: str = "1h"):
        """
        Initialize the optimizer.
        
        Args:
            strategy_type: Type of strategy to optimize
            symbol: Symbol to trade
            timeframe: Timeframe for backtesting
        """
        self.strategy_type = strategy_type
        self.symbol = symbol
        self.timeframe = timeframe
        self.results = []
        
        # Ensure strategies are loaded
        load_strategies()
        
        # Try to create a default strategy instance to get default parameters
        try:
            strategy_class = StrategyBase.get_strategy_class(strategy_type)
            self.default_params = strategy_class().parameters
            logger.info(f"Default parameters for {strategy_type}: {self.default_params}")
        except Exception as e:
            logger.warning(f"Could not get default parameters for {strategy_type}: {e}")
            self.default_params = {}
    
    async def load_market_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load market data for optimization.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            pd.DataFrame: Market data
        """
        logger.info(f"Loading market data for {self.symbol} from {start_date} to {end_date}")
        
        data = await fetch_market_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if data.empty:
            logger.error(f"No data found for {self.symbol} in the specified date range")
            return pd.DataFrame()
            
        logger.info(f"Loaded {len(data)} data points for {self.symbol}")
        return data
    
    def calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate a performance score from backtest metrics.
        
        Args:
            metrics: Backtest metrics
            
        Returns:
            float: Performance score
        """
        if not metrics:
            return 0.0
            
        # Extract key metrics
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        total_profit_pct = metrics.get('total_profit_loss_pct', 0)
        max_dd_pct = metrics.get('max_drawdown_pct', 0) or 1  # Avoid division by zero
        num_trades = metrics.get('num_trades', 0)
        
        # If there are too few trades, penalize the score
        if num_trades < 20:
            trade_factor = num_trades / 20.0
        else:
            trade_factor = 1.0
        
        # Performance score formula - weighted combination of key metrics
        score = (
            (win_rate * 0.3) + 
            (profit_factor * 20 * 0.3) + 
            (total_profit_pct * 0.3) - 
            (max_dd_pct * 0.1)
        ) * trade_factor
        
        return score
    
    async def evaluate_parameters(
        self, 
        params: Dict[str, Any],
        start_date: datetime, 
        end_date: datetime
    ) -> Tuple[Dict[str, Any], float]:
        """
        Evaluate a set of parameters using backtesting.
        
        Args:
            params: Strategy parameters
            start_date: Start date for backtesting
            end_date: End date for backtesting
            
        Returns:
            Tuple[Dict[str, Any], float]: Metrics and performance score
        """
        # Create backtest config
        config = BacktestConfig(
            strategy_type=self.strategy_type,
            strategy_params=params,
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=10000.0
        )
        
        # Run backtest
        try:
            results, metrics, trades = await backtest_engine.run_backtest(config)
            
            if not metrics:
                logger.warning(f"No metrics returned for parameter set: {params}")
                return {}, 0.0
            
            # Calculate performance score
            score = self.calculate_performance_score(metrics)
            
            # Log result
            logger.info(f"Evaluated parameters with score {score:.2f}: {params}")
            
            return metrics, score
            
        except Exception as e:
            logger.error(f"Error evaluating parameters: {str(e)}")
            return {}, 0.0
    
    async def grid_search(
        self,
        parameter_ranges: Dict[str, Any],
        start_date: datetime,
        end_date: datetime,
        max_evaluations: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Perform grid search optimization.
        
        Args:
            parameter_ranges: Parameter ranges to search
            start_date: Start date for backtesting
            end_date: End date for backtesting
            max_evaluations: Maximum number of evaluations
            
        Returns:
            List[Dict[str, Any]]: Optimization results
        """
        logger.info(f"Starting grid search for {self.strategy_type} on {self.symbol}")
        
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
        logger.info(f"Generated {total_combinations} parameter combinations for grid search")
        
        # Limit the number of evaluations if needed
        if total_combinations > max_evaluations:
            logger.warning(f"Too many combinations ({total_combinations}), limiting to {max_evaluations}")
            np.random.shuffle(param_combinations)
            param_combinations = param_combinations[:max_evaluations]
        
        # Run backtest for each parameter combination
        results = []
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            # Evaluate parameters
            metrics, score = await self.evaluate_parameters(params, start_date, end_date)
            
            if not metrics:
                continue
            
            # Save result
            result = {
                'parameters': params,
                'metrics': metrics,
                'performance': score
            }
            
            results.append(result)
            print(json.dumps(result))  # Print result as JSON
        
        # Sort results by performance score
        results.sort(key=lambda x: x['performance'], reverse=True)
        
        return results
    
    async def genetic_algorithm(
        self,
        parameter_ranges: Dict[str, Any],
        start_date: datetime,
        end_date: datetime,
        population_size: int = 20,
        generations: int = 5,
        mutation_rate: float = 0.2,
        elite_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform genetic algorithm optimization.
        
        Args:
            parameter_ranges: Parameter ranges to search
            start_date: Start date for backtesting
            end_date: End date for backtesting
            population_size: Size of the population
            generations: Number of generations
            mutation_rate: Mutation rate
            elite_size: Number of elite individuals to keep
            
        Returns:
            List[Dict[str, Any]]: Optimization results
        """
        logger.info(f"Starting genetic algorithm optimization for {self.strategy_type} on {self.symbol}")
        
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
        
        all_results = []
        
        # Evolution loop
        for generation in range(1, generations + 1):
            logger.info(f"Generation {generation}/{generations}")
            
            # Evaluate population
            generation_results = []
            fitnesses = []
            
            for i, params in enumerate(population):
                logger.info(f"Evaluating individual {i+1}/{len(population)} in generation {generation}")
                
                # Evaluate parameters
                metrics, score = await self.evaluate_parameters(params, start_date, end_date)
                fitnesses.append(score)
                
                # Save result
                result = {
                    'parameters': params,
                    'metrics': metrics,
                    'performance': score,
                    'generation': generation
                }
                
                generation_results.append(result)
                all_results.append(result)
                print(json.dumps(result))  # Print result as JSON
            
            # If last generation, skip selection and breeding
            if generation == generations:
                break
            
            # Select parents
            parent_indices = np.argsort(fitnesses)[-int(population_size/2):]  # Select best half
            parents = [population[i] for i in parent_indices]
            parent_fitnesses = [fitnesses[i] for i in parent_indices]
            
            # Create next generation
            next_population = []
            
            # Elitism - keep best individuals
            sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
            next_population.extend(sorted_population[:elite_size])
            
            # Create children through crossover and mutation
            while len(next_population) < population_size:
                # Select two parents weighted by fitness
                if sum(parent_fitnesses) > 0:
                    weights = np.array(parent_fitnesses) / sum(parent_fitnesses)
                    parent1, parent2 = np.random.choice(
                        parents, 
                        size=2, 
                        p=weights
                    )
                else:
                    # If all fitnesses are 0, select randomly
                    parent1, parent2 = np.random.choice(parents, size=2)
                
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
        
        # Sort all results by performance
        all_results.sort(key=lambda x: x['performance'], reverse=True)
        
        return all_results
    
    async def ml_optimization(
        self,
        parameter_ranges: Dict[str, Any],
        start_date: datetime,
        end_date: datetime,
        initial_samples: int = 10,
        iterations: int = 20,
        exploration_ratio: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform ML-based optimization.
        
        Args:
            parameter_ranges: Parameter ranges to search
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_samples: Number of initial random samples
            iterations: Total number of iterations
            exploration_ratio: Ratio of exploration vs. exploitation
            
        Returns:
            List[Dict[str, Any]]: Optimization results
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, falling back to random search")
            return await self._random_search(parameter_ranges, start_date, end_date, iterations)
        
        logger.info(f"Starting ML-based optimization for {self.strategy_type} on {self.symbol}")
        
        # Initialize results storage
        all_results = []
        X_samples = []
        y_samples = []
        
        # Generate initial random samples
        for i in range(initial_samples):
            logger.info(f"Generating initial sample {i+1}/{initial_samples}")
            
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
            
            # Evaluate parameters
            metrics, score = await self.evaluate_parameters(params, start_date, end_date)
            
            if not metrics:
                continue
            
            # Save result
            result = {
                'parameters': params,
                'metrics': metrics,
                'performance': score,
                'iteration': i + 1
            }
            
            results.append(result)
            print(json.dumps(result))  # Print result as JSON
        
        # Sort results by performance
        results.sort(key=lambda x: x['performance'], reverse=True)
        
        return results


async def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Strategy Optimizer")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to trade (e.g., 'BTC/USDT')")
    parser.add_argument("--strategy", type=str, required=True, help="Strategy type to optimize")
    parser.add_argument("--days", type=int, default=180, help="Number of days for historical data")
    parser.add_argument("--method", type=str, default="grid", 
                        choices=["grid", "genetic", "ml"], help="Optimization method")
    parser.add_argument("--params", type=str, help="Parameter ranges as JSON")
    parser.add_argument("--max-evaluations", type=int, default=100, help="Maximum number of evaluations for grid search")
    parser.add_argument("--population-size", type=int, default=20, help="Population size for genetic algorithm")
    parser.add_argument("--generations", type=int, default=5, help="Number of generations for genetic algorithm")
    parser.add_argument("--iterations", type=int, default=30, help="Number of iterations for ML optimization")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Parse parameter ranges
    if args.params:
        try:
            parameter_ranges = json.loads(args.params)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON for parameter ranges: {args.params}")
            return
    else:
        # Use default parameter ranges
        parameter_ranges = {}
    
    # Create optimizer
    optimizer = StrategyOptimizer(
        strategy_type=args.strategy,
        symbol=args.symbol,
        timeframe="1h"  # Default timeframe
    )
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Run optimization
    if args.method == "grid":
        results = await optimizer.grid_search(
            parameter_ranges=parameter_ranges,
            start_date=start_date,
            end_date=end_date,
            max_evaluations=args.max_evaluations
        )
    elif args.method == "genetic":
        results = await optimizer.genetic_algorithm(
            parameter_ranges=parameter_ranges,
            start_date=start_date,
            end_date=end_date,
            population_size=args.population_size,
            generations=args.generations
        )
    elif args.method == "ml":
        results = await optimizer.ml_optimization(
            parameter_ranges=parameter_ranges,
            start_date=start_date,
            end_date=end_date,
            iterations=args.iterations
        )
    
    # Print optimization report
    if results:
        best_result = results[0]  # First result is the best one
        
        report = {
            "optimization_method": args.method,
            "symbol": args.symbol,
            "strategy_type": args.strategy,
            "total_evaluations": len(results),
            "best_parameters": best_result["parameters"],
            "best_performance": best_result["performance"],
            "best_metrics": best_result["metrics"]
        }
        
        print("\n=== Optimization Report ===")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

            
            all_results.append(result)
            print(json.dumps(result))  # Print result as JSON
            
            # Create feature vector from parameters
            feature_vector = []
            for param, value_range in parameter_ranges.items():
                feature_vector.append(params.get(param, 0))
            
            X_samples.append(feature_vector)
            y_samples.append(score)
        
        # Train a model if we have enough samples
        if len(X_samples) > 3:
            model = RandomForestRegressor(n_estimators=10)
            model.fit(X_samples, y_samples)
            
            # Use the model for guided exploration
            for i in range(initial_samples, iterations):
                # Decide between exploration and exploitation
                if np.random.random() < exploration_ratio:
                    # Exploration - try random parameters
                    logger.info(f"Exploration phase - iteration {i+1}/{iterations}")
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
                    logger.info(f"Exploitation phase - iteration {i+1}/{iterations}")
                    
                    # Generate multiple random candidates and pick the one with highest predicted score
                    num_candidates = 100
                    best_candidate = None
                    best_predicted_score = -float('inf')
                    
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
                        
                        if predicted_score > best_predicted_score:
                            best_predicted_score = predicted_score
                            best_candidate = candidate_params
                    
                    params = best_candidate
                
                # Evaluate parameters
                metrics, score = await self.evaluate_parameters(params, start_date, end_date)
                
                if not metrics:
                    continue
                
                # Save result
                result = {
                    'parameters': params,
                    'metrics': metrics,
                    'performance': score,
                    'iteration': i + 1
                }
                
                all_results.append(result)
                print(json.dumps(result))  # Print result as JSON
                
                # Add to samples for ML model
                feature_vector = []
                for param, value_range in parameter_ranges.items():
                    feature_vector.append(params.get(param, 0))
                
                X_samples.append(feature_vector)
                y_samples.append(score)
                
                # Retrain model periodically
                if (i - initial_samples) % 5 == 0:
                    model.fit(X_samples, y_samples)
        
        # Sort results by performance
        all_results.sort(key=lambda x: x['performance'], reverse=True)
        
        return all_results
    
    async def _random_search(
        self,
        parameter_ranges: Dict[str, Any],
        start_date: datetime,
        end_date: datetime,
        iterations: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Perform random search optimization.
        
        Args:
            parameter_ranges: Parameter ranges to search
            start_date: Start date for backtesting
            end_date: End date for backtesting
            iterations: Number of iterations
            
        Returns:
            List[Dict[str, Any]]: Optimization results
        """
        logger.info(f"Starting random search for {self.strategy_type} on {self.symbol}")
        
        results = []
        
        for i in range(iterations):
            logger.info(f"Random search iteration {i+1}/{iterations}")
            
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
            
            # Evaluate parameters
            metrics, score = await self.evaluate_parameters(params, start_date, end_date)
            
            if not metrics:
                continue
            
            # Save result
            result = {
                'parameters': params,
                'metrics': metrics,
                'performance': score,
                'iteration': i + 1
            }
