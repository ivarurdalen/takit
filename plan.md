# Narwhals Integration Plan for takit

## Goal
Replace direct pandas usage with narwhals to support both pandas and polars (and other backends) across all indicators, signals, and the data module.

## Key Decisions
- **pandas and polars become optional** — only `narwhals` and `pydantic` are required dependencies
- **pandera is removed** — replaced with narwhals-compatible validation
- **Data module** gets an output format option (pandas/polars)
- **Approach**: `@nw.narwhalify` decorator on all public functions, manual `nw.from_native`/`nw.to_native` where needed internally

---

## Phase 1: Dependencies & Validation Layer

### 1.1 Update `pyproject.toml`
- Remove `pandas`, `pandera`, `pyarrow` from required dependencies
- Add `narwhals>=1.20` as required dependency
- Add `pandas>=2` and `polars>=1` as optional dependency groups
- Keep `pydantic>=2` as required
- Add `polars` to test dependency group

### 1.2 Replace `src/takit/validation.py`
Replace pandera schemas with narwhals-compatible validation functions:

```python
import narwhals as nw

def validate_ohlc(df: nw.DataFrame, min_length: int = 1) -> nw.DataFrame:
    """Validate OHLC DataFrame has required columns and sufficient rows."""
    required = {"open", "high", "low", "close"}
    missing = required - set(df.collect_schema().names())
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")
    if df.shape[0] <= min_length:
        raise ValueError(f"DataFrame must have more than {min_length} rows.")
    return df

def validate_series(series: nw.Series, min_length: int = 1) -> nw.Series:
    """Validate Series has sufficient length."""
    if series.len() <= min_length:
        raise ValueError(f"Series must have more than {min_length} values.")
    return series
```

Remove `@pa.check_types` decorator from `simple_moving_average`.
Remove `pandera.pandas` imports everywhere.

---

## Phase 2: Indicator Conversion (Bottom-Up)

Convert from leaf functions upward. Each function gets `@nw.narwhalify` and uses narwhals API internally.

### 2.1 `indicators/trend/ma.py` — SMA, EMA, RMA
**Current pandas ops → narwhals equivalents:**
| pandas | narwhals |
|--------|----------|
| `series.rolling(window=length).mean()` | `series.rolling_mean(window_size=length)` |
| `series.ewm(span=length, adjust=False, min_periods=mp).mean()` | `series.ewm_mean(span=length, adjust=False, min_samples=mp)` |
| `series.to_numpy()` + numpy loop + `pd.Series(rma, index=...)` | Convert to native, compute with numpy, construct back via native |
| `series.rename(...)` | `series.alias(...)` |

**RMA special case**: narwhals doesn't have a native RMA. Strategy: use `nw.to_native()` to get the native Series, compute RMA using backend-specific operations (pandas ewm or polars map), then wrap back. Alternatively, implement using `ewm_mean` with `alpha=1/length, adjust=False` which is mathematically equivalent to RMA.

### 2.2 `indicators/momentum/rsi.py`
**Key conversions:**
| pandas | narwhals |
|--------|----------|
| `series.to_frame(name="series")` | `nw.from_native(series).to_frame("series")` or use `@nw.narwhalify` |
| `df["change"] = df["series"].diff()` | `df.with_columns(change=nw.col("series").diff())` |
| `df["change"].clip(lower=0)` | `nw.col("change").clip(lower_bound=0)` |
| `df["change"].clip(upper=0).abs()` | `nw.col("change").clip(upper_bound=0).abs()` |
| `.ewm(alpha=alpha, min_periods=length).mean()` | `.ewm_mean(alpha=alpha, min_samples=length)` |
| `df.dropna(axis=0, how="any")` | `df.drop_nulls()` |
| Column arithmetic `df["a"] / df["b"]` | `nw.col("a") / nw.col("b")` |

### 2.3 `indicators/momentum/williams_r.py`
Uses OHLC DataFrame with `.rolling().max()` and `.rolling().min()` — direct narwhals equivalents exist (`rolling_mean` pattern → use expression-based approach with `with_columns`).

### 2.4 `indicators/momentum/ma_deviation.py`
Simple — just division and rename. Minimal changes beyond type annotations.

### 2.5 `indicators/momentum/ma_streak.py`
**Challenging**: Uses `.to_frame()`, `.diff()`, `.map()`, `value_streak()` (groupby + cumcount).
- `.map()` → `nw.when().then().otherwise()` for the `_wrangle_change` function
- `value_streak()` needs rewrite (see Phase 4)
- `.iloc[:, -2:]` → `.select(df.columns[-2:])`

### 2.6 `indicators/volatility/bollinger_bands.py`
| pandas | narwhals |
|--------|----------|
| `series.rolling(window=length).std(ddof=ddof)` | `series.rolling_std(window_size=length, ddof=ddof)` |
| `pd.concat(cols, axis=1)` | `nw.concat(cols, how="horizontal")` or build with `with_columns` |
| `(basis - n_std_dev * std).rename(...)` | `.alias(...)` |

### 2.7 `indicators/volatility/atr.py`
| pandas | narwhals |
|--------|----------|
| `ohlc["close"].shift(1)` | `nw.col("close").shift(1)` |
| `(expr).abs()` | `expr.abs()` |
| `ohlc[["a","b","c"]].max(axis=1)` | `nw.max_horizontal("a", "b", "c")` |

### 2.8 `indicators/volatility/donchian_channel.py`
Similar to williams_r — rolling max/min, concat horizontal.

### 2.9 `indicators/performance/relative_change.py`
- `series.loc[start:]` → need to handle filtering by index value (may need `nw.to_native()` fallback or filter expression)
- `series.iloc[0]` → `series[0]` or `series.item(0)`

---

## Phase 3: Signal Conversion

### 3.1 `signals/trend/trend_rider.py`
| pandas | narwhals |
|--------|----------|
| `np.where(cond1, 1, np.where(cond2, -1, 0))` | `nw.when(cond1).then(1).when(cond2).then(-1).otherwise(0)` |
| `df.iloc[:, -3:]` | `df.select(df.columns[-3:])` |

**Note**: narwhals doesn't support chained `.when().then().when().then()` (issue #668). Workaround: nest the when expressions or use two separate expressions combined.

### 3.2 `signals/trend/ma_cross.py`
Similar to trend_rider. Additional:
- `df[signal_name].diff().replace({2: 1, -2: -1})` → compute diff, then use `when/then/otherwise`
- `getattr(ma, fast_ma.mode)` pattern stays (Python-level dispatch)

### 3.3 `signals/volatility/williams_vix_fix.py`
- `.rolling().max()` → `rolling_max()` (if available) or expression-based
- `.rename(lambda x: ...)` → manual column renaming
- `pd.concat([...], axis=1)` → `nw.concat([...], how="horizontal")`
- `.astype(int)` → `.cast(nw.Int64)` or `.cast(nw.Int32)`

### 3.4 `signals/volatility/bb_squeeze.py`
- `(df[col] == lowest_width).astype(int)` → `nw.when(nw.col(col) == lowest_width).then(1).otherwise(0)`

### 3.5 `signals/cycles/btc.py`
Mostly DataFrame column arithmetic + calls to sma/ma_cross. Straightforward after Phase 2.

---

## Phase 4: Utility Conversion

### 4.1 `util/base.py` — `value_streak()`
**Most complex conversion**. Current implementation uses:
- `series.shift()` → `series.shift(1)` ✓
- `(series != series.shift()).cumsum()` → need cumulative sum of boolean
- `series.groupby(groups).cumcount() + 1` → no direct narwhals `cumcount`

Strategy: Implement using narwhals expressions with `cum_sum` on a boolean mask and windowed `cum_count` via `over()`, or fall back to native computation.

---

## Phase 5: Data Module

### 5.1 `data/base.py` and `data/binance_client.py`
- Keep pandas as the internal engine for data fetching (Binance API returns lists)
- Add output format parameter: `output_format: Literal["pandas", "polars"] = "pandas"`
- Use `nw.from_native()` + `nw.to_native()` for format conversion
- Move `pandas` import to be conditional (only needed when fetching data)

---

## Phase 6: Type Hints & Pydantic Integration

### 6.1 Update function signatures
- Replace `pd.Series` → `nw.Series` or use `IntoSeriesT` / `IntoFrameT` from `narwhals.typing`
- Replace `pd.DataFrame` → `nw.DataFrame` or `IntoFrameT`
- `@validate_call(config=config_dict)` — check compatibility with narwhals types. May need to adjust `arbitrary_types_allowed` or wrap inputs differently.

### 6.2 Pydantic `@validate_call` compatibility
Pydantic's `@validate_call` expects type-annotated parameters. With narwhals types:
- Option A: Use `@nw.narwhalify` *outside* `@validate_call` so pydantic sees native types
- Option B: Keep pydantic for non-DataFrame params only, handle DataFrame/Series validation manually
- **Recommended**: Stack decorators — `@nw.narwhalify` on the outside, `@validate_call` on the inside for scalar params. Remove DataFrame/Series from pydantic validation.

---

## Phase 7: Testing

- Add tests for every indicator/signal with both pandas and polars inputs
- Verify numerical output matches between backends
- Test edge cases: empty series, single-element series, NaN handling

---

## File Change Summary

| File | Changes |
|------|---------|
| `pyproject.toml` | Deps: +narwhals, -pandera, pandas/polars optional |
| `validation.py` | Full rewrite: pandera → narwhals validation |
| `indicators/trend/ma.py` | narwhals Series API, remove pandas import |
| `indicators/momentum/rsi.py` | narwhals DataFrame expressions |
| `indicators/momentum/williams_r.py` | narwhals rolling + expressions |
| `indicators/momentum/ma_deviation.py` | Minimal: type hints + alias |
| `indicators/momentum/ma_streak.py` | narwhals expressions + value_streak rewrite |
| `indicators/volatility/bollinger_bands.py` | rolling_std, concat horizontal |
| `indicators/volatility/atr.py` | shift, abs, max_horizontal |
| `indicators/volatility/donchian_channel.py` | rolling max/min, concat |
| `indicators/performance/relative_change.py` | Index filtering adaptation |
| `signals/trend/trend_rider.py` | when/then/otherwise replaces np.where |
| `signals/trend/ma_cross.py` | when/then/otherwise, diff + replace logic |
| `signals/volatility/williams_vix_fix.py` | concat, rename, cast |
| `signals/volatility/bb_squeeze.py` | when/then/otherwise for bool→int |
| `signals/cycles/btc.py` | Mostly just follows from above changes |
| `util/base.py` | value_streak rewrite with narwhals |
| `data/base.py` | Add output format parameter |
| `data/binance_client.py` | Add output format support |

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| narwhals missing operation (e.g., cumcount) | Medium | Fall back to native via `nw.to_native()` for specific operations |
| `@validate_call` incompatibility with narwhals types | Medium | Stack decorators; validate scalars only with pydantic |
| EWM parameter differences | Low | narwhals `ewm_mean` has same params (span, alpha, adjust, min_samples) |
| Index preservation (DatetimeIndex) | Medium | narwhals handles index differently than pandas; may need explicit index column |
| Chained when/then not supported | Low | Use nested when or combine expressions |
