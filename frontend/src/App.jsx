import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { createChart, ColorType, CrosshairMode, CandlestickSeries, AreaSeries, HistogramSeries } from "lightweight-charts";

const API = import.meta.env.VITE_API_URL || "/api";

/* ── Utility helpers ────────────────────────────────────────────────── */
const num = (v, d = 2) => (v == null || Number.isNaN(+v) ? "--" : (+v).toFixed(d));
const pct = (v, d = 2) => (v == null || Number.isNaN(+v) ? "--" : `${(+v).toFixed(d)}%`);
const cur = (v, c = "INR") =>
  v == null || Number.isNaN(+v)
    ? "--"
    : new Intl.NumberFormat(c === "INR" ? "en-IN" : "en-US", {
        style: "currency",
        currency: c,
        maximumFractionDigits: 2,
      }).format(v);

function fmtTs(v) {
  if (!v) return "--";
  if (typeof v === "number" && v < 1e12) v *= 1000;
  const d = new Date(v);
  return Number.isNaN(d.getTime()) ? String(v) : d.toLocaleString("en-IN", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}

const json = (p, o) =>
  fetch(`${API}${p}`, o).then(async (r) => {
    if (!r.ok) {
      const t = await r.text();
      try { throw new Error(JSON.parse(t).detail || t); } catch { throw new Error(t); }
    }
    return r.json();
  });

/* ══════════════════════════════════════════════════════════════════════
   TRADINGVIEW CHART COMPONENT — lightweight-charts v5
   ══════════════════════════════════════════════════════════════════════ */
function TVChart({ candles, ticker, chartType = "candlestick", height = 420 }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;
    if (chartRef.current) { chartRef.current.remove(); chartRef.current = null; }

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#4a5a7a",
        fontFamily: "'Share Tech Mono', monospace",
        fontSize: 10,
      },
      grid: {
        vertLines: { color: "rgba(0,255,255,0.03)" },
        horzLines: { color: "rgba(0,255,255,0.04)" },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "rgba(0,255,255,0.4)", width: 1, style: 2, labelBackgroundColor: "#0a1020" },
        horzLine: { color: "rgba(0,255,255,0.4)", width: 1, style: 2, labelBackgroundColor: "#0a1020" },
      },
      rightPriceScale: { borderColor: "rgba(0,255,255,0.08)", scaleMargins: { top: 0.08, bottom: 0.22 } },
      timeScale: { borderColor: "rgba(0,255,255,0.08)", timeVisible: false, fixLeftEdge: true, fixRightEdge: true },
      handleScroll: { mouseWheel: true, pressedMouseMove: true },
      handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    });
    chartRef.current = chart;

    if (candles?.length) {
      const sorted = [...candles].sort((a, b) => a.time - b.time);
      const deduped = [];
      const seen = new Set();
      for (let i = sorted.length - 1; i >= 0; i--) {
        if (!seen.has(sorted[i].time)) { seen.add(sorted[i].time); deduped.unshift(sorted[i]); }
      }

      if (chartType === "candlestick") {
        const series = chart.addSeries(CandlestickSeries, {
          upColor: "#00ffa3", downColor: "#ff3860",
          borderUpColor: "#00ffa3", borderDownColor: "#ff3860",
          wickUpColor: "rgba(0,255,163,0.5)", wickDownColor: "rgba(255,56,96,0.5)",
        });
        series.setData(deduped.map(c => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close })));
      } else {
        const series = chart.addSeries(AreaSeries, {
          topColor: "rgba(0,255,255,0.15)", bottomColor: "rgba(0,255,255,0.01)",
          lineColor: "#00ffff", lineWidth: 2,
          crosshairMarkerVisible: true, crosshairMarkerRadius: 4, crosshairMarkerBackgroundColor: "#00ffff",
        });
        series.setData(deduped.map(c => ({ time: c.time, value: c.close })));
      }

      const volSeries = chart.addSeries(HistogramSeries, { priceFormat: { type: "volume" }, priceScaleId: "volume" });
      chart.priceScale("volume").applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
      volSeries.setData(deduped.map(c => ({
        time: c.time, value: c.volume || 0,
        color: c.close >= c.open ? "rgba(0,255,163,0.10)" : "rgba(255,56,96,0.10)",
      })));
      chart.timeScale().fitContent();
    }

    const ro = new ResizeObserver((entries) => { for (const e of entries) chart.applyOptions({ width: e.contentRect.width }); });
    ro.observe(containerRef.current);
    return () => { ro.disconnect(); chart.remove(); chartRef.current = null; };
  }, [candles, chartType, height]);

  return (
    <div className="tv-chart-wrap">
      <div ref={containerRef} className="tv-chart-container" />
      {(!candles || !candles.length) && <div className="tv-chart-empty"><div className="spinner" /> Loading chart...</div>}
    </div>
  );
}

/* ── Small reusable components ─────────────────────────────────────── */
const Stat = ({ label, value, tone = "neutral" }) => (
  <div className={`stat tone-${tone}`}><span>{label}</span><strong>{value}</strong></div>
);

const Kpi = ({ label, value, detail, tone = "neutral" }) => (
  <div className={`kpi tone-${tone}`}><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>
);

function ConfBar({ name, score, weight }) {
  const w = Math.max(2, Math.round(score * 100));
  const t = score >= 0.6 ? "high" : score >= 0.3 ? "mid" : "low";
  return (
    <div className="conf-factor">
      <div className="conf-factor-header">
        <span className="conf-factor-name">{name}</span>
        <span className="conf-factor-meta">{(score * 100).toFixed(0)}% &times; {(weight * 100).toFixed(0)}%</span>
      </div>
      <div className="conf-bar-track"><div className={`conf-bar-fill conf-${t}`} style={{ width: `${w}%` }} /></div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════
   FUTURISTIC WALL CLOCK — IST Real-Time
   Concentric arc rings (seconds/minutes/hours) + tick marks
   Inspired by flo-bit.dev minimal precision aesthetic
   ══════════════════════════════════════════════════════════════════════ */
function WallClock() {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  // Convert to IST (UTC+5:30)
  const ist = useMemo(() => {
    return new Date(now.getTime() + (330 + now.getTimezoneOffset()) * 60000);
  }, [now]);

  const s  = ist.getSeconds();
  const ms = now.getMilliseconds(); // for smooth sub-second feel
  const m  = ist.getMinutes();
  const h  = ist.getHours();
  const h12 = h % 12;
  const weekday = ist.toLocaleDateString("en-IN", { weekday: "short" });
  const dayMonth = ist.toLocaleDateString("en-IN", { day: "2-digit", month: "short" });

  // Market session (NSE trading hours IST)
  const totalMins = h * 60 + m;
  const isOpen = totalMins >= 9 * 60 + 15 && totalMins <= 15 * 60 + 30;
  const isPreOpen = totalMins >= 9 * 60 && totalMins < 9 * 60 + 15;
  const sessionLabel = isOpen ? "NSE OPEN" : isPreOpen ? "PRE-OPEN" : "CLOSED";
  const sessionCls = isOpen ? "open" : isPreOpen ? "pre" : "closed";

  // SVG constants
  const SIZE = 110;
  const CX = SIZE / 2;  // 55
  const CY = SIZE / 2;  // 55
  const R_SEC  = 48;    // outer arc  — seconds (cyan)
  const R_MIN  = 37;    // middle arc — minutes (magenta)
  const R_HRS  = 26;    // inner arc  — hours   (electric blue)
  const TICK_INNER = 50;
  const TICK_OUTER = 54;

  const circ = (r) => 2 * Math.PI * r;

  // Arc properties: progress 0→1, starting at top (−90°)
  const arcStyle = (r, progress, color, w = 2) => {
    const c = circ(r);
    return {
      fill: "none",
      stroke: color,
      strokeWidth: w,
      strokeLinecap: "round",
      strokeDasharray: c,
      strokeDashoffset: c * (1 - progress),
      transform: `rotate(-90 ${CX} ${CY})`,
      style: { transition: "stroke-dashoffset 0.8s cubic-bezier(0.4, 0, 0.2, 1)" },
    };
  };

  const secProgress = s / 60;
  const minProgress = (m + s / 60) / 60;
  const hrsProgress = (h12 + m / 60) / 12;

  // Moving second dot position
  const secAngle = secProgress * 2 * Math.PI - Math.PI / 2;
  const dotX = CX + R_SEC * Math.cos(secAngle);
  const dotY = CY + R_SEC * Math.sin(secAngle);

  // Tick marks (60 positions)
  const ticks = useMemo(() => Array.from({ length: 60 }, (_, i) => {
    const angle = (i / 60) * 2 * Math.PI - Math.PI / 2;
    const isMajor = i % 5 === 0;
    const r0 = isMajor ? TICK_INNER - 2 : TICK_INNER;
    const r1 = isMajor ? TICK_OUTER + 1 : TICK_OUTER - 1;
    return {
      x1: CX + r0 * Math.cos(angle), y1: CY + r0 * Math.sin(angle),
      x2: CX + r1 * Math.cos(angle), y2: CY + r1 * Math.sin(angle),
      isMajor,
    };
  }), []);

  // Digital time string
  const hStr = String(h).padStart(2, "0");
  const mStr = String(m).padStart(2, "0");
  const sStr = String(s).padStart(2, "0");

  return (
    <div className="wall-clock">
      <div className="clock-svg-wrap">
        <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`}>
          {/* ── Outer boundary ring ── */}
          <circle cx={CX} cy={CY} r={TICK_OUTER + 3} fill="none" stroke="rgba(0,255,255,0.05)" strokeWidth="0.5" />

          {/* ── Tick marks ── */}
          {ticks.map((t, i) => (
            <line key={i} x1={t.x1} y1={t.y1} x2={t.x2} y2={t.y2}
              stroke={t.isMajor ? "rgba(0,255,255,0.5)" : "rgba(0,255,255,0.14)"}
              strokeWidth={t.isMajor ? 1.5 : 0.6} />
          ))}

          {/* ── Track arcs (dim background) ── */}
          <circle cx={CX} cy={CY} r={R_SEC} fill="none" stroke="rgba(0,255,255,0.06)" strokeWidth="2" />
          <circle cx={CX} cy={CY} r={R_MIN} fill="none" stroke="rgba(255,0,255,0.06)" strokeWidth="2" />
          <circle cx={CX} cy={CY} r={R_HRS} fill="none" stroke="rgba(77,127,255,0.07)" strokeWidth="3" />

          {/* ── Live arcs ── */}
          {/* Seconds — cyan */}
          <circle cx={CX} cy={CY} r={R_SEC} {...arcStyle(R_SEC, secProgress, "#00ffff", 2)} />
          {/* Minutes — magenta */}
          <circle cx={CX} cy={CY} r={R_MIN} {...arcStyle(R_MIN, minProgress, "#ff00ff", 2)} />
          {/* Hours — electric blue */}
          <circle cx={CX} cy={CY} r={R_HRS} {...arcStyle(R_HRS, hrsProgress, "#4d7fff", 3)} />

          {/* ── Center dot ── */}
          <circle cx={CX} cy={CY} r={3} fill="rgba(0,255,255,0.9)">
            <animate attributeName="r" values="2.5;3.5;2.5" dur="2s" repeatCount="indefinite" />
          </circle>
          <circle cx={CX} cy={CY} r={7} fill="none" stroke="rgba(0,255,255,0.15)" strokeWidth="1" />

          {/* ── Second dot (moves around outer arc) ── */}
          <circle cx={dotX} cy={dotY} r={3.5} fill="#00ffff">
            <filter id="sec-glow">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </circle>
          <circle cx={dotX} cy={dotY} r={5} fill="rgba(0,255,255,0.2)" />
        </svg>
      </div>

      {/* ── Digital readout ── */}
      <div className="clock-digital">
        <div className="clock-hms">
          <span className="clock-h">{hStr}</span>
          <span className="clock-sep">:</span>
          <span className="clock-m">{mStr}</span>
          <span className="clock-sep dim">:</span>
          <span className="clock-s">{sStr}</span>
        </div>
        <div className="clock-meta-row">
          <span className="clock-date-str">{weekday} {dayMonth}</span>
          <span className="clock-tz">IST</span>
        </div>
        <div className={`clock-session-badge ${sessionCls}`}>{sessionLabel}</div>
      </div>
    </div>
  );
}

function PulseMeter({ score, grade }) {
  const rotation = ((score || 50) / 100) * 180 - 90;
  const color = score >= 60 ? "var(--green)" : score >= 40 ? "var(--amber)" : "var(--red)";
  return (
    <div className="pulse-meter">
      <svg viewBox="0 0 120 70" className="pulse-arc">
        <path d="M 10 65 A 50 50 0 0 1 110 65" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="8" strokeLinecap="round" />
        <path d="M 10 65 A 50 50 0 0 1 110 65" fill="none" stroke={color} strokeWidth="8" strokeLinecap="round"
          strokeDasharray={`${(score / 100) * 157} 157`} className="pulse-arc-fill" />
        <line x1="60" y1="65" x2="60" y2="22" stroke={color} strokeWidth="2.5" strokeLinecap="round"
          transform={`rotate(${rotation}, 60, 65)`} className="pulse-needle" />
        <circle cx="60" cy="65" r="4" fill={color} />
      </svg>
      <div className="pulse-label">
        <strong style={{ color }}>{score?.toFixed(1) || "--"}</strong>
        <span>{grade || "---"}</span>
      </div>
    </div>
  );
}

function ExchangeGlobe({ exchanges, active, onSelect }) {
  const flagMap = { IN: "\ud83c\uddee\ud83c\uddf3", US: "\ud83c\uddfa\ud83c\uddf8", GB: "\ud83c\uddec\ud83c\udde7", JP: "\ud83c\uddef\ud83c\uddf5", HK: "\ud83c\udded\ud83c\uddf0", CN: "\ud83c\udde8\ud83c\uddf3", AU: "\ud83c\udde6\ud83c\uddfa", DE: "\ud83c\udde9\ud83c\uddea" };
  return (
    <div className="exchange-globe">
      {exchanges.map((ex) => (
        <button key={ex.code} className={`globe-chip ${active === ex.code ? "active" : ""}`}
          onClick={() => onSelect(ex.code)} title={`${ex.name} (${ex.country})`}>
          <span className="globe-flag">{flagMap[ex.flag] || "\ud83c\udff3\ufe0f"}</span>
          <span className="globe-code">{ex.code}</span>
        </button>
      ))}
    </div>
  );
}

function MoverCard({ item, cc = "INR" }) {
  const tone = item.percent_change >= 0 ? "buy" : "sell";
  return (
    <div className={`mover-card tone-${tone}`}>
      <div className="mover-top">
        <strong>{item.ticker?.replace(".NS", "").replace(".BO", "")}</strong>
        <span className={`mover-pct ${tone}`}>{item.percent_change >= 0 ? "+" : ""}{num(item.percent_change)}%</span>
      </div>
      <span className="mover-name">{item.name?.slice(0, 22)}</span>
    </div>
  );
}

function MarketStatusBadge({ mktStatus }) {
  if (!mktStatus) return null;
  const isOpen = mktStatus?.isOpen;
  return (
    <div className={`mkt-status ${isOpen ? "open" : "closed"}`}>
      <span className="mkt-dot" />
      <span>{isOpen ? "Market Open" : "Market Closed"}</span>
    </div>
  );
}

/* ── Tab icon SVGs ─────────────────────────────────────────────────── */
const TabIcons = {
  chart: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 12l-7-7-5 5L2 2"/><path d="M22 12V2h-10"/></svg>,
  analysis: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12a9 9 0 11-6.219-8.56"/><path d="M21 3v9h-9"/></svg>,
  markets: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10A15.3 15.3 0 0112 2z"/></svg>,
  screener: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>,
  news: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 22h16a2 2 0 002-2V4a2 2 0 00-2-2H8a2 2 0 00-2 2v16a2 2 0 01-2 2zm0 0a2 2 0 01-2-2v-9c0-1.1.9-2 2-2h2"/><path d="M18 14h-8M15 18h-5M10 6h8v4h-8z"/></svg>,
};

/* ══════════════════════════════════════════════════════════════════════ */
/*  MAIN APP                                                            */
/* ══════════════════════════════════════════════════════════════════════ */
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
  const reqRef = useRef(0);

  // Chart state
  const [candles, setCandles] = useState([]);
  const [chartType, setChartType] = useState("area");
  const [chartPeriod, setChartPeriod] = useState("1y");

  // Exchange / Pulse state
  const [pulse, setPulse] = useState(null);
  const [pulseLoading, setPulseLoading] = useState(false);
  const [exchanges, setExchanges] = useState([]);
  const [activeExchange, setActiveExchange] = useState("NSE");
  const [exchangeData, setExchangeData] = useState(null);
  const [exchangeLoading, setExchangeLoading] = useState(false);
  const [indices, setIndices] = useState([]);
  const [mktStatus, setMktStatus] = useState(null);

  // TABBED NAVIGATION
  const [activeTab, setActiveTab] = useState("chart");

  const signalTone = advisory?.signal === "BUY" ? "buy" : advisory?.signal === "SELL" ? "sell" : "hold";
  const quoteTone = (quote?.percent_change || 0) >= 0 ? "buy" : "sell";
  const newsTone = news?.average_label === "bullish" ? "buy" : news?.average_label === "bearish" ? "sell" : "neutral";
  const cc = ticker.endsWith(".NS") || ticker.endsWith(".BO") ? "INR" : "USD";
  const activeStories = useMemo(() => news?.stories || [], [news]);
  const bands = advisory?.uncertainty_bands;
  const confFactors = advisory?.confidence_factors;

  /* ── Load candle data ──────────────────────────────────────────────── */
  const loadCandles = useCallback(async (t = ticker, p = chartPeriod) => {
    try {
      const data = await json(`/candles?ticker=${encodeURIComponent(t)}&period=${encodeURIComponent(p)}`);
      setCandles(data.candles || []);
    } catch { setCandles([]); }
  }, []);

  /* ── Data Loading ─────────────────────────────────────────────────── */
  async function loadDashboard(t = ticker, p = period) {
    const rid = ++reqRef.current;
    setLoading(true); setAnalyticsLoading(true); setError("");
    setAdvisory(null); setBacktest(null); setAiBrief(null); setPulse(null);
    try {
      const [h, m, q, hi, n] = await Promise.all([
        json("/health"), json(`/model-status?ticker=${encodeURIComponent(t)}`),
        json(`/quote?ticker=${encodeURIComponent(t)}`),
        json(`/history?ticker=${encodeURIComponent(t)}&period=${encodeURIComponent(p)}`),
        json(`/news?ticker=${encodeURIComponent(t)}&limit=20`),
      ]);
      if (rid !== reqRef.current) return;
      setHealth(h); setModelStatus(m); setQuote(q); setHistory(hi); setNews(n); setLoading(false);

      loadCandles(t, chartPeriod);

      setPulseLoading(true);
      json(`/pulse?ticker=${encodeURIComponent(t)}`).then((pd) => { if (rid === reqRef.current) setPulse(pd); }).catch(() => {}).finally(() => setPulseLoading(false));

      if (m?.ready) {
        Promise.all([
          json(`/advisory?ticker=${encodeURIComponent(t)}&period=${encodeURIComponent(p)}`),
          json(`/backtest?ticker=${encodeURIComponent(t)}&period=${encodeURIComponent(p)}`),
        ]).then(([a, b]) => { if (rid === reqRef.current) { setAdvisory(a); setBacktest(b); } })
          .catch((e) => { if (rid === reqRef.current) setError((c) => c || e.message); })
          .finally(() => { if (rid === reqRef.current) setAnalyticsLoading(false); });
      } else { setAnalyticsLoading(false); }
    } catch (e) {
      if (rid === reqRef.current) { setError(e.message || "Dashboard load failed."); setAnalyticsLoading(false); }
    } finally { if (rid === reqRef.current) setLoading(false); }
  }

  async function loadScreener() {
    setScreenerLoading(true);
    try { setScreener(await json("/screener?top_n=30")); }
    catch (e) { setError((c) => c || e.message); }
    finally { setScreenerLoading(false); }
  }

  const loadExchange = useCallback(async (code) => {
    setActiveExchange(code); setExchangeLoading(true);
    try { setExchangeData(await json(`/exchange/${code}`)); }
    catch { setExchangeData(null); }
    finally { setExchangeLoading(false); }
  }, []);

  async function loadIndices() { try { setIndices(await json("/indices")); } catch { } }
  async function loadExchangeList() { try { setExchanges(await json("/exchanges")); } catch { } }
  async function loadMktStatus() { try { setMktStatus(await json("/market-status")); } catch { } }

  useEffect(() => { loadDashboard(ticker, period); }, [ticker, period]);
  useEffect(() => { loadScreener(); loadExchangeList(); loadIndices(); loadExchange("NSE"); loadMktStatus(); }, []);
  useEffect(() => { loadCandles(ticker, chartPeriod); }, [chartPeriod, ticker]);

  // Auto-refresh quote every 30s
  useEffect(() => {
    const t = setInterval(() => {
      fetch(`${API}/quote?ticker=${encodeURIComponent(ticker)}`).then(r => r.json()).then(setQuote).catch(() => {});
    }, 30000);
    return () => clearInterval(t);
  }, [ticker]);

  // Symbol search
  useEffect(() => {
    const q = tickerInput.trim();
    if (!q || q.toUpperCase() === ticker.toUpperCase()) { setSuggestions([]); return; }
    const t = setTimeout(async () => {
      try { setSuggestions((await json(`/symbols?query=${encodeURIComponent(q)}&limit=8`)).results || []); } catch { setSuggestions([]); }
    }, 250);
    return () => clearTimeout(t);
  }, [tickerInput, ticker]);

  const doTrain = async () => { setTraining(true); setError(""); try { await json(`/train?ticker=${encodeURIComponent(ticker)}&period=${encodeURIComponent(period)}&epochs=30`, { method: "POST" }); await loadDashboard(ticker, period); } catch (e) { setError(e.message); } finally { setTraining(false); } };
  const doPack = async () => { setPackTraining(true); setError(""); try { await json(`/train-starter-pack?period=${encodeURIComponent(period)}&epochs=50`, { method: "POST" }); await loadDashboard(ticker, period); } catch (e) { setError(e.message); } finally { setPackTraining(false); } };
  const doAi = async () => { setAiLoading(true); setError(""); try { setAiBrief(await json(`/ai-brief?ticker=${encodeURIComponent(ticker)}&period=${encodeURIComponent(period)}`)); } catch (e) { setError(e.message); } finally { setAiLoading(false); } };

  const TABS = [
    { id: "chart", label: "Dashboard" },
    { id: "analysis", label: "Analysis" },
    { id: "markets", label: "Markets" },
    { id: "screener", label: "Screener" },
    { id: "news", label: "News" },
  ];

  /* ── RENDER ───────────────────────────────────────────────────────── */
  return (
    <div className="app">
      {/* ▸ HEADER */}
      <header className="header">
        <div className="brand">
          <div className="logo"><span>SP</span><div className="logo-ring" /></div>
          <div>
            <p className="eyebrow accent">HOLOGRAPHIC TERMINAL</p>
            <h1>Stock Pulse <span className="version">v4.0</span></h1>
          </div>
        </div>

        {/* ── Futuristic Wall Clock ── */}
        <WallClock />

        <div className="toolbar">
          <div className="search-wrap">
            <svg className="search-icon" viewBox="0 0 20 20" fill="none"><circle cx="8.5" cy="8.5" r="5.5" stroke="currentColor" strokeWidth="1.8"/><path d="M13 13l4 4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/></svg>
            <input className="search-input" value={tickerInput} onChange={(e) => setTickerInput(e.target.value.toUpperCase())} placeholder="Search symbol..." />
            {suggestions.length > 0 && (
              <div className="suggestions">{suggestions.map((i) => (
                <button key={`${i.symbol}-${i.exchange}`} className="sug-item" onClick={() => { setTickerInput(i.symbol); setTicker(i.symbol); setSuggestions([]); }}>
                  <strong>{i.symbol}</strong><span>{i.name}</span>
                </button>
              ))}</div>
            )}
          </div>
          <select value={period} onChange={(e) => setPeriod(e.target.value)} className="select">{["1y","2y","5y"].map(p=><option key={p} value={p}>{p.toUpperCase()}</option>)}</select>
          <button className="btn primary" onClick={() => setTicker(tickerInput.trim() || "RELIANCE.NS")}>Load</button>
          <button className="btn glass" onClick={() => { loadDashboard(ticker, period); loadScreener(); loadExchange(activeExchange); loadIndices(); }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M23 4v6h-6M1 20v-6h6"/><path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/></svg>
          </button>
          <MarketStatusBadge mktStatus={mktStatus} />
        </div>
      </header>

      {error && <div className="error-bar"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/></svg>{error}</div>}

      {/* ▸ HERO — compact price bar */}
      <section className="hero-quote stagger-1">
        <div className="hero-left">
          <div className="hero-ticker-row">
            <h2 className="hero-symbol">{quote?.ticker || ticker}</h2>
            <span className={`signal-badge ${signalTone}`}>{advisory?.signal || (modelStatus?.ready ? "..." : "TRAIN")}</span>
            <MarketStatusBadge mktStatus={mktStatus} />
          </div>
          <div className="hero-price-row">
            <strong className="hero-price">{cur(quote?.price, cc)}</strong>
            <div className={`hero-change ${quoteTone}`}>
              <span>{quote?.change >= 0 ? "+" : ""}{num(quote?.change)}</span>
              <span className="hero-pct">({pct(quote?.percent_change)})</span>
            </div>
            <span className={`source-chip ${quote?.source === "Groww" ? "live" : ""}`}>{quote?.source || "Offline"}</span>
          </div>
        </div>
        <div className="hero-stats">
          <div className="hero-stat"><span>Open</span><strong>{cur(quote?.open, cc)}</strong></div>
          <div className="hero-stat"><span>High</span><strong className="buy-text">{cur(quote?.high, cc)}</strong></div>
          <div className="hero-stat"><span>Low</span><strong className="sell-text">{cur(quote?.low, cc)}</strong></div>
          <div className="hero-stat"><span>Vol</span><strong>{num(quote?.volume, 0)}</strong></div>
          <div className="hero-stat"><span>RSI</span><strong>{num(history?.latest?.rsi)}</strong></div>
          <div className="hero-stat"><span>MACD</span><strong>{num(history?.latest?.macd)}</strong></div>
        </div>
      </section>

      {/* ▸ KPI STRIP */}
      <section className="kpi-strip stagger-2">
        <Kpi label="Source" value={quote?.source || "Offline"} detail={health?.groww_configured ? "Groww active" : "Yahoo"} tone={quote?.source === "Groww" ? "buy" : "neutral"} />
        <Kpi label="Sentiment" value={news?.sentiment_method === "finbert" ? "FinBERT" : "Basic"} detail={`${news?.stats?.count ?? 0} stories`} tone={newsTone} />
        <Kpi label="Models" value={health?.trained_tickers?.length ?? 0} detail="trained" tone={modelStatus?.ready ? "buy" : "neutral"} />
        <Kpi label="Forecast" value={advisory?.signal || "--"} detail={modelStatus?.ready ? `${pct(advisory?.direction_pct)} 5d` : "train first"} tone={signalTone} />
        <Kpi label="Pulse" value={pulse ? `${pulse.score?.toFixed(0)}` : "--"} detail={pulse?.grade || "---"} tone={pulse?.score >= 60 ? "buy" : pulse?.score >= 40 ? "hold" : pulse?.score != null ? "sell" : "neutral"} />
      </section>

      {/* ▸ TAB NAVIGATION */}
      <nav className="tab-nav">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {TabIcons[tab.id]}
            <span>{tab.label}</span>
          </button>
        ))}
      </nav>

      {/* ▸ TAB CONTENT */}
      <main className="tab-content">

        {/* ═══ TAB: DASHBOARD ═══════════════════════════════════════════ */}
        {activeTab === "chart" && (
          <div className="tab-grid tab-dashboard">
            {/* Chart */}
            <section className="panel chart-panel stagger-1">
              <div className="panel-head">
                <div>
                  <p className="eyebrow muted">Price Action</p>
                  <h3>{ticker} <span className="chart-source">{candles.length ? `${candles.length} candles` : ""}</span></h3>
                </div>
                <div className="chart-controls">
                  {["1m","3m","6m","1y","2y","5y"].map(p => (
                    <button key={p} className={`chart-period-btn ${chartPeriod === p ? "active" : ""}`} onClick={() => setChartPeriod(p)}>{p.toUpperCase()}</button>
                  ))}
                  <div className="chart-type-toggle">
                    <button className={`chart-type-btn ${chartType === "area" ? "active" : ""}`} onClick={() => setChartType("area")} title="Area">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 12l-7-7-5 5L2 2"/><path d="M22 12V2h-10"/></svg>
                    </button>
                    <button className={`chart-type-btn ${chartType === "candlestick" ? "active" : ""}`} onClick={() => setChartType("candlestick")} title="Candles">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 2v4M9 18v4M15 2v7M15 17v5"/><rect x="7" y="6" width="4" height="12" rx="1"/><rect x="13" y="9" width="4" height="8" rx="1"/></svg>
                    </button>
                  </div>
                </div>
              </div>
              <TVChart candles={candles} ticker={ticker} chartType={chartType} height={420} />
              <div className="chart-meta">
                <Stat label="SMA 10" value={cur(history?.latest?.sma_10, cc)} />
                <Stat label="SMA 30" value={cur(history?.latest?.sma_30, cc)} />
                <Stat label="Vol Ratio" value={num(history?.latest?.volume_ratio)} />
                <Stat label="Data Pts" value={history?.rows || 0} />
              </div>
            </section>

            {/* Prediction side panel */}
            <section className="panel predict-panel stagger-2">
              <div className="panel-head">
                <div><p className="eyebrow muted">LSTM Forecast</p><h3>{advisory?.signal || "--"}</h3></div>
                <span className={`signal-pill ${signalTone}`}>{modelStatus?.ready ? pct(advisory?.direction_pct) : "idle"}</span>
              </div>
              {modelStatus?.ready ? (
                analyticsLoading ? <div className="empty-state sm"><div className="spinner" /> Loading...</div> : (
                <>
                  {/* Stale-model warning — shown when live price drifts >10% from training data */}
                  {advisory?.model_stale_warning && (
                    <div className="warn-bar amber">{advisory.model_stale_warning}</div>
                  )}
                  <div className="pred-grid">
                    {(advisory?.predictions || []).map((item, idx) => (
                      <div key={item.day} className={`pred-card ${idx === (advisory?.predictions?.length || 0) - 1 ? "final" : ""}`}>
                        <span className="pred-day">Day +{item.day}</span>
                        <strong className="pred-price">{cur(item.price, cc)}</strong>
                        <small className={item.pct_change >= 0 ? "buy-text" : "sell-text"}>{pct(item.pct_change)}</small>
                        {bands && <span className="band">{cur(bands.lower?.[idx], cc)} — {cur(bands.upper?.[idx], cc)}</span>}
                      </div>
                    ))}
                  </div>
                  {/* Confidence */}
                  <div className="conf-section">
                    <div className="conf-row"><span className="conf-label">Confidence</span><strong className={`tone-${signalTone}`}>{pct(advisory?.confidence)}</strong></div>
                    {confFactors && <div className="conf-factors">{Object.entries(confFactors).map(([k, v]) => <ConfBar key={k} name={k.replace(/_/g, " ")} score={v.score} weight={v.weight} />)}</div>}
                  </div>
                  {/* Explanation */}
                  {advisory?.explanation && (
                    <div className="expl-section">
                      <div className="expl-head"><strong>{advisory.explanation.headline}</strong><span className={`signal-pill sm ${advisory.explanation.confluence === "strong" ? "buy" : advisory.explanation.confluence === "moderate" ? "hold" : "sell"}`}>{advisory.explanation.confluence}</span></div>
                      {advisory.explanation.reasoning?.length > 0 && <ul className="expl-reasons">{advisory.explanation.reasoning.map((r, i) => <li key={i}>{r}</li>)}</ul>}
                      {advisory.explanation.technicals?.length > 0 && <div className="expl-techs">{advisory.explanation.technicals.map((t, i) => <span key={i} className="tech-chip">{t}</span>)}</div>}
                    </div>
                  )}
                  {/* Action buttons */}
                  <div className="panel-actions">
                    <button className="btn accent-btn" onClick={doAi} disabled={aiLoading}>{aiLoading ? "Thinking..." : "AI Brief"}</button>
                    <button className="btn" onClick={doPack} disabled={packTraining}>{packTraining ? "Training..." : "Starter Pack"}</button>
                  </div>
                </>
              )) : (
                <div className="empty-state stacked">
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--muted)" strokeWidth="1.5"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
                  <span>Train {ticker} to unlock LSTM forecasts</span>
                  <button className="btn primary" onClick={doTrain} disabled={training}>{training ? "Training..." : "Train Model"}</button>
                </div>
              )}
            </section>
          </div>
        )}

        {/* ═══ TAB: ANALYSIS ═══════════════════════════════════════════ */}
        {activeTab === "analysis" && (
          <div className="tab-grid tab-analysis">
            {/* Backtest */}
            <section className="panel bt-panel stagger-1">
              <div className="panel-head"><div><p className="eyebrow muted">Backtest</p><h3>Risk & Return</h3></div>{backtest?.threshold_type && <span className="badge">{backtest.threshold_type}</span>}</div>
              {modelStatus?.ready ? (analyticsLoading ? <div className="empty-state sm"><div className="spinner" /> Loading backtest...</div> : (
                <>
                  {backtest?.metrics?.sample_warning && <div className="warn-bar">{backtest.metrics.sample_warning}</div>}
                  <div className="bt-grid">
                    <Stat label="Return" value={pct(backtest?.metrics?.total_return_pct)} tone="buy" />
                    <Stat label="Sharpe" value={num(backtest?.metrics?.sharpe, 3)} />
                    <Stat label="Max DD" value={pct(backtest?.metrics?.max_drawdown_pct)} tone="sell" />
                    <Stat label="Win Rate" value={pct(backtest?.metrics?.win_rate_pct)} />
                    <Stat label="Trades" value={backtest?.metrics?.total_trades ?? "--"} />
                    <Stat label="Capital" value={cur(backtest?.metrics?.final_capital, cc)} />
                    <Stat label="PF" value={num(backtest?.metrics?.profit_factor, 2)} />
                    <Stat label="Sortino" value={num(backtest?.metrics?.sortino, 3)} />
                  </div>
                  <div className="bt-quality">
                    <p className="eyebrow muted" style={{ marginBottom: 8 }}>Prediction Quality</p>
                    <div className="bt-grid">
                      <Stat label="Dir. Accuracy" value={pct(backtest?.metrics?.directional_accuracy_pct)} tone={backtest?.metrics?.directional_accuracy_pct > 55 ? "buy" : "neutral"} />
                      <Stat label="Info Coeff." value={num(backtest?.metrics?.information_coefficient, 4)} tone={backtest?.metrics?.information_coefficient > 0.05 ? "buy" : "neutral"} />
                      <Stat label="Predictions" value={backtest?.metrics?.n_predictions ?? "--"} />
                      <Stat label="Stat. Sig." value={backtest?.metrics?.stat_significant ? "Yes" : "No"} tone={backtest?.metrics?.stat_significant ? "buy" : "sell"} />
                    </div>
                  </div>
                  {backtest?.stat_conclusion && <div className={`stat-conclusion ${backtest?.metrics?.stat_significant ? "sig-yes" : "sig-no"}`}>{backtest.stat_conclusion}</div>}
                  <div className="trade-list">{(backtest?.recent_trades || []).slice(0, 8).map((t, i) => (
                    <div key={i} className="trade-row"><span>{t.entry_date}</span><span className="trade-arrow">&rarr;</span><span>{t.exit_date}</span>{t.direction && <span className="trade-dir">{t.direction}</span>}<strong className={t.pnl_pct >= 0 ? "buy-text" : "sell-text"}>{pct(t.pnl_pct)}</strong></div>
                  ))}</div>
                </>
              )) : <div className="empty-state sm">Train {ticker} to unlock backtest.</div>}
            </section>

            {/* Alpha Pulse Engine */}
            <section className="panel pulse-panel stagger-2">
              <div className="panel-head"><div><p className="eyebrow accent">Alpha Pulse Engine</p><h3>APE Score</h3></div><span className="badge glow">{pulse?.grade || "---"}</span></div>
              {pulseLoading ? <div className="empty-state sm"><div className="spinner" /> Computing pulse &amp; AI assessment...</div> : pulse ? (
                <>
                  <PulseMeter score={pulse.score} grade={pulse.grade} />
                  {/* Show score source: fused or technical-only */}
                  {pulse.raw_technical_score != null && (
                    <div className="pulse-fusion-bar">
                      <span className="fusion-label">Technical: {pulse.raw_technical_score?.toFixed(0)}</span>
                      <span className="fusion-sep">×</span>
                      <span className="fusion-label accent">AI: {pulse.ai_assessment?.llm_score?.toFixed(0) ?? "--"}</span>
                      <span className="fusion-sep">→</span>
                      <span className="fusion-label glow">Fused: {pulse.score?.toFixed(0)}</span>
                    </div>
                  )}
                  <div className="pulse-dims">
                    {pulse.dimensions && Object.entries(pulse.dimensions).map(([k, v]) => (
                      <div key={k} className="pulse-dim">
                        <span>{k.replace(/_/g, " ")}</span>
                        <div className="dim-bar-track"><div className={`dim-bar-fill ${v >= 0.6 ? "high" : v >= 0.4 ? "mid" : "low"}`} style={{ width: `${v * 100}%` }} /></div>
                        <strong>{(v * 100).toFixed(0)}</strong>
                      </div>
                    ))}
                  </div>
                  {/* AI Assessment Section */}
                  {pulse.ai_assessment && (
                    <div className="ai-assess-block">
                      <div className="ai-assess-head">
                        <span className="eyebrow accent">AI Intelligence</span>
                        <span className={`signal-pill sm ${pulse.ai_assessment.llm_grade === "STRONG BUY" || pulse.ai_assessment.llm_grade === "BUY" ? "buy" : pulse.ai_assessment.llm_grade === "HOLD" ? "hold" : "sell"}`}>
                          {pulse.ai_assessment.llm_grade}
                        </span>
                      </div>
                      {pulse.ai_assessment.reasoning && (
                        <div className="ai-reasoning">
                          {pulse.ai_assessment.reasoning.split("|").map((r, i) => (
                            <div key={i} className="reason-line">• {r.trim()}</div>
                          ))}
                        </div>
                      )}
                      {pulse.ai_assessment.catalysts && (
                        <div className="ai-catalysts">
                          <span className="eyebrow muted">Catalysts &amp; Risks</span>
                          {pulse.ai_assessment.catalysts.split("|").map((c, i) => (
                            <div key={i} className="catalyst-chip">{c.trim()}</div>
                          ))}
                        </div>
                      )}
                      {pulse.ai_assessment.action && (
                        <div className="ai-action-line">
                          <strong>Action:</strong> {pulse.ai_assessment.action}
                        </div>
                      )}
                    </div>
                  )}
                  {/* LSTM Forecast Summary */}
                  {pulse.lstm_forecast && (
                    <div className="pulse-lstm-row">
                      <Stat label="LSTM Signal" value={pulse.lstm_forecast.signal} />
                      <Stat label="Confidence" value={pct(pulse.lstm_forecast.confidence)} />
                      <Stat label="5d Move" value={pct(pulse.lstm_forecast.direction_pct)} />
                    </div>
                  )}
                  {/* Fundamentals Summary */}
                  {pulse.fundamentals && (
                    <div className="pulse-fund-row">
                      <Stat label="Sector" value={pulse.fundamentals.sector} />
                      <Stat label="P/E" value={num(pulse.fundamentals.pe_ratio, 1)} />
                      <Stat label="ROE" value={pct(pulse.fundamentals.roe_pct, 1)} />
                      <Stat label="D/E" value={num(pulse.fundamentals.debt_to_equity, 2)} />
                    </div>
                  )}
                  {pulse.alerts?.length > 0 && <div className="pulse-alerts">{pulse.alerts.map((a, i) => <div key={i} className="alert-chip">{a}</div>)}</div>}
                  <div className="pulse-footer">
                    <Stat label="Regime" value={pulse.regime} />
                    <Stat label="Confluence" value={pct(pulse.confluence * 100)} />
                    <Stat label="Momentum" value={pulse.momentum_direction} />
                  </div>
                </>
              ) : <div className="empty-state sm">Load a ticker to compute.</div>}
            </section>

            {/* AI Brief */}
            <section className="panel ai-panel stagger-3">
              <div className="panel-head"><div><p className="eyebrow accent">AI Analysis</p><h3>{health?.gemini_configured ? "Gemini 2.5" : "Offline"}</h3></div><span className="badge">{aiBrief?.model || (health?.gemini_configured ? "ready" : "no key")}</span></div>
              {aiBrief?.summary ? <div className="ai-copy">{aiBrief.summary}</div> : (
                <div className="empty-state sm">
                  {health?.gemini_configured ? (
                    <><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--magenta)" strokeWidth="1.5"><path d="M12 2a10 10 0 100 20 10 10 0 000-20z"/><path d="M12 6v6l4 2"/></svg><span>Click <strong>AI Brief</strong> for Gemini analysis</span>
                    <button className="btn accent-btn" onClick={doAi} disabled={aiLoading} style={{ marginTop: 8 }}>{aiLoading ? "Thinking..." : "Generate AI Brief"}</button></>
                  ) : "Gemini key not set."}
                </div>
              )}
            </section>
          </div>
        )}

        {/* ═══ TAB: MARKETS ═══════════════════════════════════════════ */}
        {activeTab === "markets" && (
          <div className="tab-grid tab-markets">
            {/* Global Indices */}
            {indices.length > 0 && (
              <section className="panel indices-panel stagger-1">
                <div className="panel-head"><div><p className="eyebrow accent">Global Indices</p><h3>Real-time Benchmarks</h3></div><span className="badge">{indices.length} indices</span></div>
                <div className="indices-grid">
                  {indices.map((idx, i) => (
                    <div key={`${idx.exchange}-${i}`} className="index-card" onClick={() => loadExchange(idx.exchange)}>
                      <span className="idx-name">{idx.name}</span>
                      {idx.value != null ? (
                        <>
                          <strong>{num(idx.value, 0)}</strong>
                          <span className={idx.percent_change >= 0 ? "buy-text" : "sell-text"}>{idx.percent_change >= 0 ? "+" : ""}{num(idx.percent_change)}%</span>
                        </>
                      ) : <span className="muted">--</span>}
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Exchange Explorer */}
            <section className="panel exchange-panel stagger-2">
              <div className="panel-head"><div><p className="eyebrow accent">Exchange Explorer</p><h3>Gainers & Losers</h3></div><span className="badge">{exchanges.length} exchanges</span></div>
              <ExchangeGlobe exchanges={exchanges} active={activeExchange} onSelect={loadExchange} />
              {exchangeLoading ? <div className="empty-state sm"><div className="spinner" /> Loading exchange data...</div> : exchangeData ? (
                <>
                  <div className="exchange-header">
                    <div className="exchange-title"><strong>{exchangeData.name}</strong><span className="muted"> &middot; {exchangeData.country} &middot; {exchangeData.currency}</span></div>
                    {exchangeData.index && (
                      <div className="exchange-index">
                        <span>{exchangeData.index.name}</span>
                        <strong>{num(exchangeData.index.value, 0)}</strong>
                        <span className={exchangeData.index.percent_change >= 0 ? "buy-text" : "sell-text"}>{exchangeData.index.percent_change >= 0 ? "+" : ""}{num(exchangeData.index.percent_change)}%</span>
                      </div>
                    )}
                  </div>
                  <div className="movers-grid">
                    <div className="movers-col">
                      <p className="movers-heading buy-text">Top Gainers</p>
                      {(exchangeData.gainers || []).map((g) => <MoverCard key={g.ticker} item={g} cc={exchangeData.currency} />)}
                    </div>
                    <div className="movers-col">
                      <p className="movers-heading sell-text">Top Losers</p>
                      {(exchangeData.losers || []).map((l) => <MoverCard key={l.ticker} item={l} cc={exchangeData.currency} />)}
                    </div>
                  </div>
                </>
              ) : <div className="empty-state sm">Select an exchange above.</div>}
            </section>
          </div>
        )}

        {/* ═══ TAB: SCREENER ══════════════════════════════════════════ */}
        {activeTab === "screener" && (
          <div className="tab-grid tab-screener">
            <section className="panel screener-panel stagger-1">
              <div className="panel-head"><div><p className="eyebrow muted">AI Stock Screener</p><h3>Indian Universe</h3></div><span className="badge">{screenerLoading ? "scanning..." : `${screener?.count || 0} tracked`}</span></div>
              {(screener?.results || []).length ? (
                <>
                  <div className="table-wrap"><table className="compact-table"><thead><tr><th>#</th><th>Symbol</th><th>Score</th><th>Fund</th><th>Tech</th><th>Mom</th><th>ROE</th><th>LSTM</th></tr></thead><tbody>
                    {(screener?.results || []).map((i) => (
                      <tr key={i.ticker}><td className="rank-cell">{i.rank}</td><td><button className="inline-link" onClick={() => { setTickerInput(i.ticker); setTicker(i.ticker); setActiveTab("chart"); }}>{i.ticker}</button></td>
                        <td><span className="score-pill">{num(i.score, 3)}</span></td><td>{num(i.fundamental, 2)}</td><td>{num(i.technical, 2)}</td><td>{num(i.momentum, 2)}</td><td>{pct(i.roe_pct)}</td>
                        <td><span className={`lstm-ind ${i.has_lstm_model ? "on" : ""}`}>{i.has_lstm_model ? "TRAINED" : "--"}</span></td>
                      </tr>
                    ))}
                  </tbody></table></div>
                  {screener?.model_coverage && <div className="coverage-note">{screener.model_coverage.disclosure}</div>}
                </>
              ) : <div className="empty-state sm">{screenerLoading ? <><div className="spinner" /> Scanning markets...</> : "No results."}</div>}
            </section>
          </div>
        )}

        {/* ═══ TAB: NEWS ══════════════════════════════════════════════ */}
        {activeTab === "news" && (
          <div className="tab-grid tab-news">
            <section className="panel news-panel-full stagger-1">
              <div className="panel-head">
                <div><p className="eyebrow muted">News Feed</p><h3>{news?.sentiment_method === "finbert" ? "FinBERT NLP Sentiment" : "News"}</h3></div>
                <span className={`signal-pill sm ${newsTone}`}>{news?.average_label || "neutral"}</span>
              </div>
              <div className="news-stats-row">
                <div className="news-stat"><span className="ns-dot bull" /><strong>{news?.stats?.bullish ?? 0}</strong><span>Bullish</span></div>
                <div className="news-stat"><span className="ns-dot bear" /><strong>{news?.stats?.bearish ?? 0}</strong><span>Bearish</span></div>
                <div className="news-stat"><span className="ns-dot neut" /><strong>{news?.stats?.neutral ?? 0}</strong><span>Neutral</span></div>
                {news?.sentiment_method && <div className="finbert-badge"><span className="dot-pulse" /> {news.sentiment_method === "finbert" ? "FinBERT Active" : news.sentiment_method}</div>}
              </div>
              <div className="news-grid">
                {activeStories.length > 0 ? activeStories.map((s, i) => (
                  <a key={`${s.title}-${i}`} className="news-card" href={s.url} target="_blank" rel="noreferrer">
                    <div className="news-top">
                      <span className={`news-dot ${s.sentiment_label || "neutral"}`} />
                      <span className="news-source">{s.source || "Source"}</span>
                      {s.sentiment_score != null && <span className={`sent-score ${s.sentiment_label}`}>{s.sentiment_score > 0 ? "+" : ""}{num(s.sentiment_score)}</span>}
                    </div>
                    <strong className="news-title">{s.title}</strong>
                    <p>{s.summary?.slice(0, 120) || "Read full article..."}</p>
                  </a>
                )) : <div className="empty-state sm">No stories available.</div>}
              </div>
            </section>
          </div>
        )}

      </main>

      {/* ▸ FOOTER */}
      <footer className="footer">
        <span>{quote?.source || "Yahoo"} / {news?.sentiment_method === "finbert" ? "FinBERT" : "Fallback"} / Finnhub</span>
        <span>LSTM {health?.device || "cpu"} | Alpha Pulse Engine v4.0</span>
        <span>PyTorch + FastAPI + React + TradingView | Holographic Terminal</span>
      </footer>
    </div>
  );
}
