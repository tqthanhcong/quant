# Quant System: Strategy-Independent Backtesting and Signal Generation

## Overview
A modular system for backtesting and live signal generation. Strategies are pure mappers from features to trade intents; engines consume intents without changing. Data is keyed by (time, symbol) with base OHLCV and derived features.

## Core Principles
- Decoupled: Strategy code does not modify engines.
- Feature-first: Strategies operate on a unified feature frame.
- Deterministic: Same features + params → same intents.
- Pluggable: Swap data, features, strategies, or engines independently.

## Data Model
- Primary key: time, symbol
- Base features: open, high, low, close, volume
- Derived features: computed columns (e.g., returns, MA, RSI)
- Signal features: buy, sell, position
- FeatureFrame: tabular, aligned, time-indexed per symbol

```mermaid
graph TD
    A[Market Data Sources] --> B[Ingestor]
    B --> C[(Time-Series Store<br>key: time,symbol<br>cols: OHLCV)]
    C --> D[Feature Pipeline]
    D --> E[[FeatureFrame]]
    subgraph strategies [Strategies]
        F[Strategy<br>pure mapping features to intents]
    end
    E --> F
    F --> G[Signal Engine]
    subgraph execution_and_backtest [Execution and Backtest]
        G --> H[Backtest Engine]
        H --> I[Broker Simulator]
        I --> J[Positions/Trades]
        J --> K[Metrics & Reports]
    end
    style F fill:#e7f5ff,stroke:#4c6ef5
    style G fill:#e6fcf5,stroke:#0ca678
    style H fill:#fff3bf,stroke:#f08c00
    style I fill:#ffe8cc,stroke:#e8590c
```

## Invariance Guarantee
- Changing a strategy only changes the mapping from features to intents.
- Engines (signal, backtest, broker sim) are unaffected and require no code changes.
- New features or data sources remain compatible as long as they conform to the FeatureFrame schema.