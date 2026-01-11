import React from "react";
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, TimeScale, TimeSeriesScale, Tooltip, Legend, PointElement, LineElement, ScatterController, type ChartOptions } from "chart.js";
import zoomPlugin from "chartjs-plugin-zoom";
import "chartjs-adapter-date-fns";
import { Chart } from "react-chartjs-2";
import { CandlestickController, CandlestickElement } from "chartjs-chart-financial";

ChartJS.register(TimeScale, TimeSeriesScale, LinearScale, BarElement, CategoryScale, PointElement, LineElement, ScatterController, CandlestickController, CandlestickElement, Tooltip, Legend, zoomPlugin);

const API_BASE: string = "http://localhost:8000/marketdata";

type HistoricalRow = {
	open_time: number | string;
	open_btc: number;
	high_btc: number;
	low_btc: number;
	close_btc: number;
} & Record<string, number | string | null>;

type CandlePoint = {
	x: number;
	o: number;
	h: number;
	l: number;
	c: number;
};

type MetricPoint = {
	x: number;
	y: number;
};

type MetricKey = "close_btc" | "volume_btc" | "close_eth" | "volume_eth" | "close_bnb" | "volume_bnb" | "close_xrp" | "volume_xrp" | "close_spx" | "volume_spx" | "gdp" | "unrate";

const METRICS: { key: MetricKey; label: string }[] = [
	{ key: "close_btc", label: "BTC - close" },
	{ key: "volume_btc", label: "BTC - volume" },
	{ key: "close_eth", label: "ETH - close" },
	{ key: "volume_eth", label: "ETH - volume" },
	{ key: "close_bnb", label: "BNB - close" },
	{ key: "volume_bnb", label: "BNB - volume" },
	{ key: "close_xrp", label: "XRP - close" },
	{ key: "volume_xrp", label: "XRP - volume" },
	{ key: "close_spx", label: "S&P 500 - close_spx" },
	{ key: "volume_spx", label: "S&P 500 - volume_spx" },
	{ key: "gdp", label: "GDP (USA)" },
	{ key: "unrate", label: "Unemployment rate (USA)" },
];

type ModelKey = "baseline" | "linear-regression" | "random-forest" | "lstm" | "tft";

const MODEL_ENDPOINT: Record<ModelKey, string> = {
	baseline: "naive-model",
	"linear-regression": "linear-regression-model",
	"random-forest": "random-forest-model",
	lstm: "lstm-model",
	tft: "tft-model",
};

function parseTimestamp(value: number | string): number {
	if (typeof value === "number") return value;
	return new Date(value).getTime();
}

function buildCandleSeries(rows: HistoricalRow[]): CandlePoint[] {
	return rows
		.map((row) => ({
			x: parseTimestamp(row.open_time),
			o: Number(row.open_btc),
			h: Number(row.high_btc),
			l: Number(row.low_btc),
			c: Number(row.close_btc),
		}))
		.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.o) && Number.isFinite(point.h) && Number.isFinite(point.l) && Number.isFinite(point.c));
}

function buildMetricSeries(rows: HistoricalRow[], metricKey: MetricKey): MetricPoint[] {
	return rows
		.map((row) => {
			const x = parseTimestamp(row.open_time);
			const raw = row[metricKey];
			const y = typeof raw === "string" ? Number(raw) : (raw as number);
			return { x, y };
		})
		.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
}

type BuildZoomOptionsParams = {
	syncEnabled: boolean;
	getTargetChart: () => ChartJS | null;
	syncingRef: React.MutableRefObject<boolean>;
};

function buildZoomOptions(params: BuildZoomOptionsParams) {
	const { syncEnabled, getTargetChart, syncingRef } = params;

	function handleSync(ctx: { chart: any }) {
		if (!syncEnabled) return;
		if (syncingRef.current) return;

		const sourceChart = ctx.chart as any;
		const scale = sourceChart.scales?.x;
		if (!scale) return;

		const min = scale.min;
		const max = scale.max;
		if (typeof min !== "number" || typeof max !== "number") return;

		const target = getTargetChart();
		if (!target) return;

		syncingRef.current = true;
		try {
			(target as any).zoomScale("x", { min, max }, "none");
		} finally {
			syncingRef.current = false;
		}
	}

	return {
		pan: {
			enabled: true,
			mode: "x" as const,
			onPan: handleSync,
		},
		zoom: {
			mode: "x" as const,
			wheel: {
				enabled: true,
			},
			pinch: {
				enabled: true,
			},
			onZoom: handleSync,
		},
		limits: {
			x: { min: "original" as const, max: "original" as const },
		},
	};
}

type BinanceCandleChartProps = {
	candles: CandlePoint[];
	zoomOptions: any;
	chartRef: React.RefObject<ChartJS<"candlestick">>;
	predictionPoint?: MetricPoint | null;
};

function BinanceCandleChart({ candles, zoomOptions, chartRef, predictionPoint }: BinanceCandleChartProps) {
	const showPrediction = predictionPoint && Number.isFinite(predictionPoint.x) && Number.isFinite(predictionPoint.y);

	const data: any = {
		datasets: [
			{
				label: "BTCUSDC - świece dzienne",
				type: "candlestick" as const,
				data: candles,
			},
			...(showPrediction
				? [
						{
							label: "Predykcja close_btc",
							type: "scatter" as const,
							parsing: false,
							data: [{ x: predictionPoint!.x, y: predictionPoint!.y }],
							pointRadius: 6,
							pointHoverRadius: 8,
							backgroundColor: "rgba(15, 118, 110, 0.95)",
							borderColor: "rgba(15, 118, 110, 0.95)",
						},
				  ]
				: []),
		],
	};

	const options: ChartOptions<"candlestick"> = {
		responsive: true,
		maintainAspectRatio: false,
		animation: false,
		transitions: {
			zoom: { animation: { duration: 0 } },
			pan: { animation: { duration: 0 } },
		},
		scales: {
			x: {
				type: "time",
				time: { unit: "day" },
			},
			y: { beginAtZero: false },
		},
		plugins: {
			legend: { position: "top" },
			tooltip: { mode: "index", intersect: false },
			zoom: zoomOptions as any,
		},
	};

	return (
		<div className="rounded-2xl border bg-white p-6 shadow-sm">
			<h4 className="font-medium mb-4">Bitcoin (świece dzienne, Binance)</h4>
			<div className="h-64">
				<Chart ref={chartRef} type="candlestick" data={data} options={options} />
			</div>
		</div>
	);
}

type MetricBarChartProps = {
	series: MetricPoint[];
	metricKey: MetricKey;
	zoomOptions: any;
	chartRef: React.RefObject<ChartJS<"bar">>;
	onMetricChange: (key: MetricKey) => void;
};

function MetricBarChart({ series, metricKey, zoomOptions, chartRef, onMetricChange }: MetricBarChartProps) {
	const metricMeta = METRICS.find((m) => m.key === metricKey);

	const data = {
		datasets: [
			{
				label: metricMeta?.label ?? metricKey,
				type: "bar" as const,
				data: series.map((p) => ({ x: p.x, y: p.y })),
				backgroundColor: "rgba(88, 80, 241, 0.85)",
			},
		],
	};

	const options: ChartOptions<"bar"> = {
		responsive: true,
		maintainAspectRatio: false,
		animation: false,
		transitions: {
			zoom: { animation: { duration: 0 } },
			pan: { animation: { duration: 0 } },
		},
		parsing: false,
		scales: {
			x: {
				type: "time",
				time: { unit: "day" },
			},
			y: {
				beginAtZero: false,
			},
		},
		plugins: {
			legend: {
				position: "top",
			},
			tooltip: {
				mode: "index",
				intersect: false,
			},
			zoom: zoomOptions as any,
		},
	};

	return (
		<div className="rounded-2xl border bg-white p-6 shadow-sm">
			<div className="flex flex-wrap items-center justify-between gap-3 mb-4">
				<h4 className="font-medium">Drugi wykres (dowolny parametr z endpointu)</h4>
				<select className="rounded-xl border px-3 py-1 text-sm bg-white" value={metricKey} onChange={(e) => onMetricChange(e.target.value as MetricKey)}>
					{METRICS.map((m) => (
						<option key={m.key} value={m.key}>
							{m.label}
						</option>
					))}
				</select>
			</div>

			<div className="h-64">
				<Chart ref={chartRef} type="bar" data={data as any} options={options} />
			</div>
		</div>
	);
}

export default function Analysis() {
	const [rows, setRows] = React.useState<HistoricalRow[]>([]);
	const [loading, setLoading] = React.useState(false);
	const [error, setError] = React.useState<string | null>(null);

	const [metricKey, setMetricKey] = React.useState<MetricKey>("close_btc");
	const [selectedModel, setSelectedModel] = React.useState<ModelKey>("baseline");
	const [syncZoom, setSyncZoom] = React.useState(true);

	const [modelLoading, setModelLoading] = React.useState(false);
	const [modelError, setModelError] = React.useState<string | null>(null);
	const [predictionPoint, setPredictionPoint] = React.useState<MetricPoint | null>(null);
	const [predictedClose, setPredictedClose] = React.useState<number | null>(null);

	const candleChartRef = React.useRef<ChartJS<"candlestick"> | null>(null);
	const metricChartRef = React.useRef<ChartJS<"bar"> | null>(null);
	const syncingRef = React.useRef(false);

	React.useEffect(() => {
		let cancelled = false;

		async function load() {
			setLoading(true);
			setError(null);

			try {
				const res = await fetch(`${API_BASE}/historical-data/?years_back=2`);
				if (!res.ok) throw new Error(`HTTP ${res.status}`);
				const json = await res.json();
				if (!Array.isArray(json)) throw new Error("Nieoczekiwany format odpowiedzi z API");

				if (!cancelled) setRows(json as HistoricalRow[]);
			} catch (e: any) {
				if (!cancelled) setError(e?.message || "Błąd podczas pobierania danych");
			} finally {
				if (!cancelled) setLoading(false);
			}
		}

		load();
		return () => {
			cancelled = true;
		};
	}, []);

	const candles = React.useMemo(() => buildCandleSeries(rows), [rows]);
	const metricSeries = React.useMemo(() => buildMetricSeries(rows, metricKey), [rows, metricKey]);

	const candleZoomOptions = React.useMemo(
		() =>
			buildZoomOptions({
				syncEnabled: syncZoom,
				getTargetChart: () => metricChartRef.current as unknown as ChartJS,
				syncingRef,
			}),
		[syncZoom]
	);

	const metricZoomOptions = React.useMemo(
		() =>
			buildZoomOptions({
				syncEnabled: syncZoom,
				getTargetChart: () => candleChartRef.current as unknown as ChartJS,
				syncingRef,
			}),
		[syncZoom]
	);

	async function handleRunModel() {
		setModelLoading(true);
		setModelError(null);

		try {
			if (!rows.length) throw new Error("Brak danych historycznych - nie mogę osadzić punktu predykcji na osi czasu.");

			const endpoint = MODEL_ENDPOINT[selectedModel];
			const res = await fetch(`${API_BASE}/${endpoint}/`);
			if (!res.ok) throw new Error(`Model API HTTP ${res.status}`);

			const json = await res.json();
			const close = Number(json?.close_btc);

			if (!Number.isFinite(close)) {
				throw new Error("API modelu zwróciło niepoprawne close_btc");
			}

			const lastTs = parseTimestamp(rows[rows.length - 1].open_time);
			const nextDayTs = lastTs + 24 * 60 * 60 * 1000;

			setPredictedClose(close);
			setPredictionPoint({ x: nextDayTs, y: close });

			setMetricKey("close_btc");
		} catch (e: any) {
			setModelError(e?.message || "Błąd podczas uruchamiania modelu");
		} finally {
			setModelLoading(false);
		}
	}

	return (
		<section className="grid gap-6">
			<h2 className="text-2xl font-bold">Analiza</h2>

			<div className="rounded-2xl border bg-white p-6 shadow-sm flex flex-wrap items-center justify-between gap-4">
				<div className="flex flex-col gap-2">
					<div className="flex flex-wrap items-center gap-3 text-sm">
						<div className="flex items-center gap-2">
							<span className="text-slate-600">Model ML:</span>
							<select className="rounded-xl border px-3 py-1 bg-white" value={selectedModel} onChange={(e) => setSelectedModel(e.target.value as ModelKey)}>
								<option value="baseline">Model bazowy (naive)</option>
								<option value="linear-regression">Linear Regression</option>
								<option value="random-forest">Random Forest</option>
								<option value="lstm">LSTM</option>
								<option value="tft">TFT</option>
							</select>
						</div>

						<label className="inline-flex items-center gap-2 text-sm">
							<input type="checkbox" className="h-4 w-4" checked={syncZoom} onChange={(e) => setSyncZoom(e.target.checked)} />
							<span>Synchronizuj zoom obu wykresów</span>
						</label>
					</div>

					{predictedClose !== null && (
						<p className="text-sm text-slate-600">
							Predykcja <span className="font-medium">close_btc</span>:{" "}
							<span className="font-semibold text-slate-900">
								{predictedClose.toLocaleString(undefined, {
									maximumFractionDigits: 2,
								})}
							</span>
						</p>
					)}
					{modelError && <p className="text-sm text-red-600">{modelError}</p>}
				</div>

				<button type="button" onClick={handleRunModel} disabled={modelLoading} className="inline-flex items-center rounded-xl bg-slate-900 px-4 py-1.5 text-sm font-medium text-white hover:bg-slate-800 transition disabled:opacity-60">
					{modelLoading ? "Uruchamiam..." : "Uruchom model"}
				</button>
			</div>

			{loading && <p className="text-sm text-slate-500">Ładowanie danych historycznych...</p>}
			{error && <p className="text-sm text-red-600">{error}</p>}
			{!loading && !error && rows.length === 0 && <p className="text-sm text-slate-500">Brak danych z endpointu historical-data.</p>}

			{!loading && !error && rows.length > 0 && (
				<div className="grid gap-6">
					<BinanceCandleChart candles={candles} zoomOptions={candleZoomOptions} chartRef={candleChartRef} predictionPoint={predictionPoint} />
					<MetricBarChart series={metricSeries} metricKey={metricKey} zoomOptions={metricZoomOptions} chartRef={metricChartRef} onMetricChange={setMetricKey} />
				</div>
			)}
		</section>
	);
}
