import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "/api";

function number(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

function pct(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${Number(value).toFixed(digits)}%`;
}

function currency(value, currencyCode = "INR") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: currencyCode,
    maximumFractionDigits: 2
  }).format(value);
}

function formatTimestamp(value) {
  if (!value) return "--";
  if (typeof value === "number" && value < 1000000000000) {
    value *= 1000;
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit"
  });
}

function Sparkline({ data, dataKey, color = "#15b26b" }) {
  if (!data?.length) return <div className="empty-state">No chart data</div>;
  const values = data.map((item) => Number(item[dataKey]));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const points = values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * 100;
      const y = 100 - ((value - min) / range) * 100;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg className="sparkline" viewBox="0 0 100 100" preserveAspectRatio="none">
      <defs>
        <linearGradient id={`chart-fill-${dataKey}`} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.35" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <polygon fill={`url(#chart-fill-${dataKey})`} points={`0,100 ${points} 100,100`} />
      <polyline fill="none" stroke={color} strokeWidth="2.4" points={points} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function Stat({ label, value, tone = "neutral" }) {
  return (
    <div className={`stat tone-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MiniKpi({ label, value, detail, tone = "neutral" }) {
  return (
    <div className={`mini-kpi tone-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

/* --- Confidence Factor Bar Component --- */
function ConfidenceBar({ name, score, weight }) {
  const barWidth = Math.max(2, Math.round(score * 100));
  const tone = score >= 0.6 ? "high" : score >= 0.3 ? "mid" : "low";
  return (
    <div className="conf-factor">
      <div className="conf-factor-header">
        <span className="conf-factor-name">{name}</span>
        <span className="conf-factor-meta">{(score * 100).toFixed(0)}% &times; {(weight * 100).toFixed(0)}%</span>
      </div>
      <div className="conf-bar-track">
        <div className={`conf-bar-fill conf-${tone}`} style={{ width: `${barWidth}%` }} />
      </div>
    </div>
  );
}

/* --- Explanation Section --- */
function SignalExplanation({ explanation }) {
  if (!explanation) return null;
  const confluenceTone =
    explanation.confluence === "strong" ? "buy" :
    explanation.confluence === "moderate" ? "hold" :
    explanation.confluence === "conflicting" ? "sell" : "neutral";

  return (
    <div className="explanation-section">
      <div className="expl-header">
        <strong className="expl-headline">{explanation.headline}</strong>
        <span className={`signal-chip ${confluenceTone}`}>{explanation.confluence}</span>
      </div>
      {explanation.reasoning?.length > 0 && (
        <ul className="expl-reasons">
          {explanation.reasoning.map((reason, i) => (
            <li key={i}>{reason}</li>
          ))}
        </ul>
      )}
      {explanation.technicals?.length > 0 && (
        <div className="expl-technicals">
          {explanation.technicals.map((tech, i) => (
            <span key={i} className="tech-chip">{tech}</span>
          ))}
        </div>
      )}
      {explanation.watch_items?.length > 0 && (
        <div className="expl-watch">
          <span className="expl-watch-label">Watch:</span>
          {explanation.watch_items.map((item, i) => (
            <span key={i} className="watch-chip">{item}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function fetchJson(path, options) {
  return fetch(`${API_BASE}${path}`, options).then(async (response) => {
    if (!response.ok) {
      const text = await response.text();
      try {
        const payload = JSON.parse(text);
        throw new Error(payload.detail || text);
      } catch {
        throw new Error(text);
      }
    }
    return response.json();
  });
}

export default function App() {
  const [tickerInput, setTickerInput] = useState("RELIANCE.NS");
  const [ticker, setTicker] = useState("RELIANCE.NS");
  const [period, setPeriod] = useState("5y");
  const [health, setHealth] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [quote, setQuote] = useState(null);
  const [history, setHistory] = useState(null);
  const [advisory, setAdvisory] = useState(null);
  const [backtest, setBacktest] = useState(null);
  const [news, setNews] = useState(null);
  const [screener, setScreener] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [analyticsLoading, setAnalyticsLoading] = useState(true);
  const [screenerLoading, setScreenerLoading] = useState(true);
  const [aiBrief, setAiBrief] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [packTraining, setPackTraining] = useState(false);
  const [error, setError] = useState("");
  const requestIdRef = useRef(0);

  const signalTone = advisory?.signal === "BUY" ? "buy" : advisory?.signal === "SELL" ? "sell" : "hold";
  const quoteTone = (quote?.percent_change || 0) >= 0 ? "buy" : "sell";
  const newsTone =
    news?.average_label === "bullish" ? "buy" : news?.average_label === "bearish" ? "sell" : "neutral";
  const providerTone = quote?.source === "Groww" ? "buy" : "neutral";
  const currencyCode = ticker.endsWith(".NS") ? "INR" : "USD";

  const activeStories = useMemo(() => news?.stories || [], [news]);
  const watchlist = useMemo(() => screener?.results?.slice(0, 8) || [], [screener]);

  async function loadDashboard(activeTicker = ticker, activePeriod = period) {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    setLoading(true);
    setAnalyticsLoading(true);
    setError("");
    setAdvisory(null);
    setBacktest(null);
    setAiBrief(null);
    try {
      const [healthData, modelData, quoteData, historyData, newsData] =
        await Promise.all([
          fetchJson("/health"),
          fetchJson(`/model-status?ticker=${encodeURIComponent(activeTicker)}`),
          fetchJson(`/quote?ticker=${encodeURIComponent(activeTicker)}`),
          fetchJson(`/history?ticker=${encodeURIComponent(activeTicker)}&period=${encodeURIComponent(activePeriod)}`),
          fetchJson(`/news?ticker=${encodeURIComponent(activeTicker)}&limit=8`)
        ]);

      if (requestId !== requestIdRef.current) {
        return;
      }

      setHealth(healthData);
      setModelStatus(modelData);
      setQuote(quoteData);
      setHistory(historyData);
      setNews(newsData);
      setLoading(false);

      if (modelData?.ready) {
        Promise.all([
          fetchJson(`/advisory?ticker=${encodeURIComponent(activeTicker)}&period=${encodeURIComponent(activePeriod)}`),
          fetchJson(`/backtest?ticker=${encodeURIComponent(activeTicker)}&period=${encodeURIComponent(activePeriod)}`)
        ])
          .then(([advisoryData, backtestData]) => {
            if (requestId !== requestIdRef.current) {
              return;
            }
            setAdvisory(advisoryData);
            setBacktest(backtestData);
          })
          .catch((err) => {
            if (requestId !== requestIdRef.current) {
              return;
            }
            setError((current) => current || err.message || "Unable to load analytics.");
          })
          .finally(() => {
            if (requestId === requestIdRef.current) {
              setAnalyticsLoading(false);
            }
          });
      } else {
        setAnalyticsLoading(false);
      }
    } catch (err) {
      if (requestId !== requestIdRef.current) {
        return;
      }
      setError(err.message || "Unable to load dashboard.");
      setAnalyticsLoading(false);
    } finally {
      if (requestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  }

  async function loadScreenerData() {
    setScreenerLoading(true);
    try {
      const data = await fetchJson("/screener?top_n=12");
      setScreener(data);
    } catch (err) {
      setError((current) => current || err.message || "Unable to load screener.");
    } finally {
      setScreenerLoading(false);
    }
  }

  async function triggerTraining() {
    setTraining(true);
    setError("");
    try {
      await fetchJson(`/train?ticker=${encodeURIComponent(ticker)}&period=${encodeURIComponent(period)}&epochs=30`, {
        method: "POST"
      });
      await loadDashboard(ticker, period);
    } catch (err) {
      setError(err.message || "Training failed.");
    } finally {
      setTraining(false);
    }
  }

  async function triggerStarterPackTraining() {
    setPackTraining(true);
    setError("");
    try {
      await fetchJson(`/train-starter-pack?period=${encodeURIComponent(period)}&epochs=20`, {
        method: "POST"
      });
      await loadDashboard(ticker, period);
    } catch (err) {
      setError(err.message || "Starter pack training failed.");
    } finally {
      setPackTraining(false);
    }
  }

  async function loadAiBrief(activeTicker = ticker, activePeriod = period) {
    setAiLoading(true);
    setError("");
    try {
      const data = await fetchJson(`/ai-brief?ticker=${encodeURIComponent(activeTicker)}&period=${encodeURIComponent(activePeriod)}`);
      setAiBrief(data);
    } catch (err) {
      setError(err.message || "AI brief failed.");
    } finally {
      setAiLoading(false);
    }
  }

  async function refreshQuote(activeTicker = ticker) {
    try {
      const data = await fetchJson(`/quote?ticker=${encodeURIComponent(activeTicker)}`);
      setQuote(data);
    } catch {
      // Keep the last quote on screen if the refresh fails.
    }
  }

  useEffect(() => {
    loadDashboard(ticker, period);
  }, [ticker, period]);

  useEffect(() => {
    loadScreenerData();
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      refreshQuote(ticker);
    }, 30000);
    return () => window.clearInterval(timer);
  }, [ticker]);

  useEffect(() => {
    const query = tickerInput.trim();
    if (!query || query.toUpperCase() === ticker.toUpperCase()) {
      setSuggestions([]);
      return;
    }

    const timer = window.setTimeout(async () => {
      try {
        const data = await fetchJson(`/symbols?query=${encodeURIComponent(query)}&limit=8`);
        setSuggestions(data.results || []);
      } catch {
        setSuggestions([]);
      }
    }, 250);

    return () => window.clearTimeout(timer);
  }, [tickerInput, ticker]);

  // Uncertainty bands for prediction cards
  const bands = advisory?.uncertainty_bands;
  const confFactors = advisory?.confidence_factors;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <span className="brand-mark">SP</span>
          <div>
            <p className="eyebrow">Prediction Terminal</p>
            <h1>Stock Pulse Dashboard</h1>
          </div>
        </div>

        <div className="toolbar">
          <div className="search-wrap">
            <input
              className="search-input"
              value={tickerInput}
              onChange={(e) => setTickerInput(e.target.value.toUpperCase())}
              placeholder="Search symbol or company"
            />
            {suggestions.length > 0 ? (
              <div className="suggestions">
                {suggestions.map((item) => (
                  <button
                    key={`${item.symbol}-${item.exchange}`}
                    className="suggestion-item"
                    onClick={() => {
                      setTickerInput(item.symbol);
                      setTicker(item.symbol);
                      setSuggestions([]);
                    }}
                  >
                    <strong>{item.symbol}</strong>
                    <span>{item.name}</span>
                  </button>
                ))}
              </div>
            ) : null}
          </div>

          <select value={period} onChange={(e) => setPeriod(e.target.value)} className="period-select">
            <option value="1y">1Y</option>
            <option value="2y">2Y</option>
            <option value="5y">5Y</option>
          </select>

          <button className="action-button primary" onClick={() => setTicker(tickerInput.trim() || "RELIANCE.NS")}>
            Load
          </button>
          <button className="action-button" onClick={() => {
            loadDashboard(ticker, period);
            loadScreenerData();
          }}>
            Refresh
          </button>
          <button className="action-button" onClick={() => loadAiBrief(ticker, period)} disabled={aiLoading}>
            {aiLoading ? "Thinking..." : "AI Brief"}
          </button>
          <button className="action-button" onClick={triggerStarterPackTraining} disabled={packTraining}>
            {packTraining ? "Warming pack..." : "Warm Starter Pack"}
          </button>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="quote-strip">
        <div className="quote-main">
          <div>
            <p className="eyebrow muted">Now Tracking</p>
            <h2>{quote?.ticker || ticker}</h2>
            <p className="subtle">{quote?.name || "Loading quote..."}</p>
            <div className="meta-line">
              <span className={`signal-chip ${providerTone}`}>{quote?.source || "Fallback"}</span>
              <span className="meta-text">Updated {formatTimestamp(quote?.timestamp)}</span>
              <span className="meta-text">{modelStatus?.ready ? "Model ready" : "Needs training"}</span>
            </div>
          </div>
          <div className="quote-price-block">
            <strong>{currency(quote?.price, currencyCode)}</strong>
            <span className={`quote-change ${quoteTone}`}>{pct(quote?.percent_change)}</span>
          </div>
        </div>

        <div className="quote-stats">
          <Stat label="Open" value={currency(quote?.open, currencyCode)} />
          <Stat label="High" value={currency(quote?.high, currencyCode)} />
          <Stat label="Low" value={currency(quote?.low, currencyCode)} />
          <Stat label="Volume" value={number(quote?.volume, 0)} />
            <Stat label="Signal" value={advisory?.signal || "train"} tone={signalTone} />
            <Stat label="News Mood" value={news?.average_label || "--"} tone={newsTone} />
            <Stat label="API" value={quote?.source || "fallback"} />
            <Stat label="Model" value={modelStatus?.ready ? "ready" : "train"} tone={modelStatus?.ready ? "buy" : "sell"} />
        </div>
      </section>

      <section className="overview-strip">
        <MiniKpi
          label="Live Feed"
          value={quote?.source || "Offline"}
          detail={health?.groww_configured ? "Groww key loaded" : "Fallback provider active"}
          tone={providerTone}
        />
        <MiniKpi
          label="Sentiment Engine"
          value={news?.sentiment_method === "finbert" ? "FinBERT" : news?.source || "Waiting"}
          detail={`${news?.stats?.count ?? 0} stories analyzed`}
          tone={newsTone}
        />
        <MiniKpi
          label="Model Library"
          value={health?.trained_tickers?.length ?? 0}
          detail="trained tickers"
          tone={modelStatus?.ready ? "buy" : "neutral"}
        />
        <MiniKpi
          label="Forecast"
          value={advisory?.signal || "--"}
          detail={modelStatus?.ready ? `${pct(advisory?.direction_pct)} in 5d` : "train this ticker to unlock"}
          tone={signalTone}
        />
      </section>

      <main className="terminal-grid">
        {/* ---- PRICE CHART PANEL ---- */}
        <section className="panel chart-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow muted">Price Action</p>
              <h3>{ticker} history</h3>
            </div>
            <span className="panel-badge">{history?.rows || 0} rows</span>
          </div>
          {loading && !(history?.history || []).length ? (
            <div className="empty-state">Loading market data...</div>
          ) : (
            <Sparkline data={history?.history || []} dataKey="close" />
          )}
          <div className="chart-meta">
            <Stat label="RSI" value={number(history?.latest?.rsi)} />
            <Stat label="MACD" value={number(history?.latest?.macd)} />
            <Stat label="SMA 30" value={currency(history?.latest?.sma_30, currencyCode)} />
            <Stat label="Volume Ratio" value={number(history?.latest?.volume_ratio)} />
          </div>
        </section>

        {/* ---- PREDICTION + EXPLAINABILITY PANEL ---- */}
        <section className="panel prediction-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow muted">Prediction</p>
              <h3>{advisory?.signal || "--"}</h3>
            </div>
            <span className={`signal-chip ${signalTone}`}>{modelStatus?.ready ? pct(advisory?.direction_pct) : "model idle"}</span>
          </div>
          {modelStatus?.ready ? (
            analyticsLoading ? (
              <div className="empty-state dense">Loading forecast...</div>
            ) : (
            <>
              {/* Prediction cards with uncertainty bands */}
              <div className="prediction-grid">
                {(advisory?.predictions || []).map((item, idx) => (
                  <div key={item.day} className="prediction-card">
                    <span>Day +{item.day}</span>
                    <strong>{currency(item.price, currencyCode)}</strong>
                    <small className={item.pct_change >= 0 ? "buy-text" : "sell-text"}>
                      {pct(item.pct_change)}
                    </small>
                    {bands && (
                      <span className="band-range">
                        {currency(bands.lower?.[idx], currencyCode)} — {currency(bands.upper?.[idx], currencyCode)}
                      </span>
                    )}
                  </div>
                ))}
              </div>

              {/* Confidence Breakdown */}
              <div className="confidence-section">
                <div className="conf-header-row">
                  <span className="conf-label">Confidence</span>
                  <strong className={`conf-value tone-${signalTone}`}>{pct(advisory?.confidence)}</strong>
                </div>
                {confFactors && (
                  <div className="conf-factors">
                    {Object.entries(confFactors).map(([key, val]) => (
                      <ConfidenceBar
                        key={key}
                        name={key.replace(/_/g, " ")}
                        score={val.score}
                        weight={val.weight}
                      />
                    ))}
                  </div>
                )}
                {advisory?.confidence_formula && (
                  <div className="conf-formula">{advisory.confidence_formula}</div>
                )}
              </div>

              {/* Signal Explainability */}
              <SignalExplanation explanation={advisory?.explanation} />

              <div className="micro-row">
                <Stat label="Focus Days" value={(advisory?.focus_days_ago || []).join(", ") || "--"} />
                <Stat label="Threshold" value={pct(advisory?.threshold_pct)} />
              </div>
            </>
            )
          ) : (
            <div className="empty-state stacked">
              <span>Train a dedicated model for {ticker} to unlock forecasts and backtests.</span>
              <button className="action-button primary" onClick={triggerTraining} disabled={training}>
                {training ? "Training..." : "Train 30 epochs"}
              </button>
            </div>
          )}
        </section>

        {/* ---- NEWS PANEL with FinBERT sentiment ---- */}
        <aside className="panel compact-panel news-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow muted">News Pulse</p>
              <h3>{news?.sentiment_method === "finbert" ? "FinBERT" : news?.source || "News"}</h3>
            </div>
            <span className={`signal-chip ${newsTone}`}>{news?.average_label || "neutral"}</span>
          </div>
          <div className="micro-row news-summary">
            <Stat label="Stories" value={news?.stats?.count ?? 0} />
            <Stat label="Bullish" value={news?.stats?.bullish ?? 0} tone="buy" />
            <Stat label="Bearish" value={news?.stats?.bearish ?? 0} tone="sell" />
          </div>
          {news?.sentiment_method && (
            <div className="sentiment-method-badge">
              <span className="method-dot" /> Powered by {news.sentiment_method === "finbert" ? "FinBERT NLP" : news.sentiment_method}
            </div>
          )}
          <div className="news-feed">
            {activeStories.length > 0 ? (
              activeStories.map((story, index) => (
                <a key={`${story.title}-${index}`} className="news-item" href={story.url} target="_blank" rel="noreferrer">
                  <div className="news-topline">
                    <span className={`news-dot ${story.sentiment_label || "neutral"}`} />
                    <span>{story.source || "Source"}</span>
                    {story.sentiment_score != null && (
                      <span className={`sentiment-score ${story.sentiment_label || "neutral"}`}>
                        {story.sentiment_score > 0 ? "+" : ""}{number(story.sentiment_score, 2)}
                      </span>
                    )}
                  </div>
                  <strong>{story.title}</strong>
                  <p>{story.summary || "Open article for the latest coverage."}</p>
                </a>
              ))
            ) : (
              <div className="empty-state dense">No fresh stories available for this ticker right now.</div>
            )}
          </div>
        </aside>

        {/* ---- BACKTEST PANEL with new ML metrics ---- */}
        <section className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow muted">Backtest Snapshot</p>
              <h3>Risk and return</h3>
            </div>
            {backtest?.threshold_type && (
              <span className="panel-badge">{backtest.threshold_type} threshold</span>
            )}
          </div>
          {modelStatus?.ready ? (
            analyticsLoading ? (
              <div className="empty-state dense">Loading backtest...</div>
            ) : (
            <>
              {/* Sample warning banner */}
              {backtest?.metrics?.sample_warning && (
                <div className="sample-warning">{backtest.metrics.sample_warning}</div>
              )}

              {/* Core financial metrics */}
              <div className="mini-terminal">
                <Stat label="Return" value={pct(backtest?.metrics?.total_return_pct)} tone="buy" />
                <Stat label="Sharpe" value={number(backtest?.metrics?.sharpe, 3)} />
                <Stat label="Max DD" value={pct(backtest?.metrics?.max_drawdown_pct)} tone="sell" />
                <Stat label="Win Rate" value={pct(backtest?.metrics?.win_rate_pct)} />
                <Stat label="Trades" value={backtest?.metrics?.total_trades ?? "--"} />
                <Stat label="Final Capital" value={currency(backtest?.metrics?.final_capital, currencyCode)} />
                <Stat label="Profit Factor" value={number(backtest?.metrics?.profit_factor, 2)} />
                <Stat label="Sortino" value={number(backtest?.metrics?.sortino, 3)} />
              </div>

              {/* Prediction quality metrics */}
              <div className="ml-metrics-section">
                <p className="eyebrow muted metrics-section-label">Prediction Quality</p>
                <div className="ml-metrics-grid">
                  <Stat
                    label="Dir. Accuracy"
                    value={pct(backtest?.metrics?.directional_accuracy_pct)}
                    tone={backtest?.metrics?.directional_accuracy_pct > 55 ? "buy" : "neutral"}
                  />
                  <Stat
                    label="Info Coefficient"
                    value={number(backtest?.metrics?.information_coefficient, 4)}
                    tone={backtest?.metrics?.information_coefficient > 0.05 ? "buy" : "neutral"}
                  />
                  <Stat
                    label="Predictions"
                    value={backtest?.metrics?.n_predictions ?? "--"}
                  />
                  <Stat
                    label="Statistical Sig."
                    value={backtest?.metrics?.stat_significant ? "Yes" : "No"}
                    tone={backtest?.metrics?.stat_significant ? "buy" : "sell"}
                  />
                </div>
              </div>

              {/* Statistical conclusion */}
              {backtest?.stat_conclusion && (
                <div className={`stat-conclusion ${backtest?.metrics?.stat_significant ? "sig-yes" : "sig-no"}`}>
                  {backtest.stat_conclusion}
                </div>
              )}

              <div className="trade-list">
                {(backtest?.recent_trades || []).slice(0, 6).map((trade, index) => (
                  <div key={`${trade.entry_date}-${trade.exit_date}-${index}`} className="trade-row">
                    <span>{trade.entry_date}</span>
                    <span>{trade.exit_date}</span>
                    <strong className={trade.pnl_pct >= 0 ? "buy-text" : "sell-text"}>{pct(trade.pnl_pct)}</strong>
                  </div>
                ))}
              </div>
            </>
            )
          ) : (
            <div className="empty-state">
              No trained artifact found for {ticker}. Quotes, search, and news remain live.
            </div>
          )}
        </section>

        {/* ---- WATCHLIST ---- */}
        <aside className="panel compact-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow muted">Market Leaders</p>
              <h3>Watchlist</h3>
            </div>
          </div>
          <div className="watchlist">
            {watchlist.map((item) => (
              <button
                key={item.ticker}
                className="watch-item"
                onClick={() => {
                  setTickerInput(item.ticker);
                  setTicker(item.ticker);
                }}
              >
                <span>{item.ticker}</span>
                <div className="watch-meta">
                  <strong>{number(item.score, 3)}</strong>
                  {item.has_lstm_model && <span className="lstm-dot" title="LSTM model trained" />}
                </div>
              </button>
            ))}
          </div>
        </aside>

        {/* ---- SCREENER PANEL with model coverage ---- */}
        <section className="panel screener-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow muted">Screener</p>
              <h3>Top opportunities</h3>
            </div>
            <span className="panel-badge">{screenerLoading ? "scanning..." : `${screener?.count || 0} tracked`}</span>
          </div>
          {(screener?.results || []).length ? (
            <>
              <div className="table-wrap">
                <table className="compact-table">
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Score</th>
                      <th>Fund</th>
                      <th>Tech</th>
                      <th>Mom</th>
                      <th>ROE</th>
                      <th>LSTM</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(screener?.results || []).map((item) => (
                      <tr key={item.ticker}>
                        <td>
                          <button
                            className="inline-link"
                            onClick={() => {
                              setTickerInput(item.ticker);
                              setTicker(item.ticker);
                            }}
                          >
                            {item.ticker}
                          </button>
                        </td>
                        <td>{number(item.score, 3)}</td>
                        <td>{number(item.fundamental, 2)}</td>
                        <td>{number(item.technical, 2)}</td>
                        <td>{number(item.momentum, 2)}</td>
                        <td>{pct(item.roe_pct)}</td>
                        <td>
                          <span className={`lstm-indicator ${item.has_lstm_model ? "active" : ""}`}>
                            {item.has_lstm_model ? "trained" : "--"}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {/* Model coverage disclosure */}
              {screener?.model_coverage && (
                <div className="model-coverage-note">
                  {screener.model_coverage.disclosure}
                </div>
              )}
            </>
          ) : (
            <div className="empty-state dense">{screenerLoading ? "Scanning the market universe..." : "No screener results yet."}</div>
          )}
        </section>

        {/* ---- AI BRIEF ---- */}
        <section className="panel ai-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow muted">AI Brief</p>
              <h3>{health?.gemini_configured ? "Gemini analysis" : "AI offline"}</h3>
            </div>
            <span className="panel-badge">{aiBrief?.model || (health?.gemini_configured ? "ready" : "no key")}</span>
          </div>
          {aiBrief?.summary ? (
            <div className="ai-copy">{aiBrief.summary}</div>
          ) : (
            <div className="empty-state dense">
              {health?.gemini_configured ? "Click AI Brief for a live summary of the selected stock." : "Gemini key not configured yet."}
            </div>
          )}
        </section>
      </main>

      <footer className="footer-bar">
        <span>Data: {quote?.source || "Yahoo Finance fallback"} / {news?.sentiment_method === "finbert" ? "FinBERT NLP" : news?.source || "Yahoo Finance fallback"}</span>
        <span>
          API {health?.api || "offline"} {health?.frontend_ready ? "| dashboard ready" : "| build missing"}{" "}
          {health?.groww_configured ? "| Groww armed" : "| Groww key missing"}{" "}
          {health?.gemini_configured ? "| Gemini armed" : "| Gemini key missing"}
        </span>
        <span>{health?.device || "cpu"}</span>
      </footer>
    </div>
  );
}
