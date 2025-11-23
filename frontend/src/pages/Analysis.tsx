import React from "react";
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, TimeScale, TimeSeriesScale, Tooltip, Legend, type ChartOptions } from "chart.js";
import "chartjs-adapter-date-fns";
import { Chart } from "react-chartjs-2";
import { CandlestickController, CandlestickElement } from "chartjs-chart-financial";

ChartJS.register(TimeScale, TimeSeriesScale, LinearScale, BarElement, CategoryScale, CandlestickController, CandlestickElement, Tooltip, Legend);

type Endpoint = { key: string; label: string };

const API_BASE = "http://localhost:8000/marketdata";

const ENDPOINTS: Endpoint[] = [
	{ key: "binance-test", label: "Binance" },
	{ key: "fred-test", label: "FRED" },
	{ key: "stooq-test", label: "Stooq" },
	{ key: "coinmetrics-test", label: "CoinMetrics" },
];

// Dane do wykresów

type ChartSeriesKey = "binance" | "fred" | "stooq" | "coinmetrics";
type BarSeriesKey = "fred" | "stooq" | "coinmetrics";

const BAR_KEYS: BarSeriesKey[] = ["fred", "stooq", "coinmetrics"];

type CandlePoint = {
	x: number; // timestamp w ms
	o: number;
	h: number;
	l: number;
	c: number;
};

type ChartSeriesValue = { type: "candles"; data: CandlePoint[] } | { type: "bars"; data: number[] };

const CHART_SERIES_LABELS: Record<ChartSeriesKey, string> = {
	binance: "Binance - świeczki BTC (przykład)",
	fred: "FRED - przykładowe dane",
	stooq: "Stooq - przykładowe dane",
	coinmetrics: "CoinMetrics - przykładowe dane",
};

// Dane przykładowe
const BINANCE_CANDLES: CandlePoint[] = [
	{
		x: new Date("2025-01-01").getTime(),
		o: 42000.5,
		h: 43010,
		l: 41850,
		c: 42890.2,
	},
	{
		x: new Date("2025-01-02").getTime(),
		o: 42890.2,
		h: 43500,
		l: 42500.5,
		c: 43210,
	},
	{
		x: new Date("2025-01-03").getTime(),
		o: 43210,
		h: 43750,
		l: 43000,
		c: 43580.7,
	},
	{
		x: new Date("2025-01-04").getTime(),
		o: 43580.7,
		h: 44020,
		l: 43310,
		c: 43850.1,
	},
	{
		x: new Date("2025-01-05").getTime(),
		o: 43850.1,
		h: 44500,
		l: 43600,
		c: 44210.4,
	},
];

const CHART_SERIES: Record<ChartSeriesKey, ChartSeriesValue> = {
	binance: {
		type: "candles",
		data: BINANCE_CANDLES,
	},
	fred: {
		type: "bars",
		data: [5, 14, 9, 12, 3, 10, 7],
	},
	stooq: {
		type: "bars",
		data: [3, 6, 9, 12, 9, 6, 3],
	},
	coinmetrics: {
		type: "bars",
		data: [10, 11, 8, 15, 13, 9, 12],
	},
};

const ML_MODELS = [
	{ key: "baseline", label: "Model bazowy" },
	{ key: "lstm", label: "LSTM" },
	{ key: "random-forest", label: "Random Forest" },
];

function ResponseView({ value }: { value: any }) {
	if (value === null || value === undefined) return null;
	if (typeof value === "string") {
		return <pre className="mt-3 max-h-[480px] overflow-auto rounded-xl bg-slate-900 p-4 text-slate-100 text-sm">{value}</pre>;
	}
	return <pre className="mt-3 max-h-[480px] overflow-auto rounded-xl bg-slate-900 p-4 text-slate-100 text-sm">{JSON.stringify(value, null, 2)}</pre>;
}

// Wykres btc
function BinanceCandleChart() {
	const series = CHART_SERIES.binance;
	if (series.type !== "candles") return null;

	const candles = series.data;
	console.log("Candles data:", candles);

	const data = {
		datasets: [
			{
				label: "BTCUSDT (Binance, przykładowe dane)",
				type: "candlestick" as const,
				data: candles,
				backgroundColor: "rgba(34, 197, 94, 0.85)",
			},
		],
	};

	const options: ChartOptions<"candlestick"> = {
		responsive: true,
		maintainAspectRatio: false,
		plugins: {
			legend: {
				position: "top",
			},
			tooltip: {
				mode: "index",
				intersect: false,
			},
		},
	};

	return (
		<div className="rounded-2xl border bg-white p-6 shadow-sm">
			<h4 className="font-medium mb-4">Bitcoin (świeczki)</h4>
			<div className="h-64">
				<Chart type="candlestick" data={data} options={options} />
			</div>
		</div>
	);
}

// Wykres słupkowy

type BarChartCardProps = {
	title: string;
	selectedKey: BarSeriesKey;
	onChangeKey: (key: BarSeriesKey) => void;
};

function BarChartCard({ title, selectedKey, onChangeKey }: BarChartCardProps) {
	const series = CHART_SERIES[selectedKey];
	if (series.type !== "bars") return null;

	const values = series.data;
	const labels = values.map((_, i) => `${i + 1}`);

	const data = {
		labels,
		datasets: [
			{
				label: CHART_SERIES_LABELS[selectedKey],
				data: values,
				backgroundColor: "rgba(88, 80, 241, 0.85)",
			},
		],
	};

	const options: ChartOptions<"bar"> = {
		responsive: true,
		maintainAspectRatio: false,
		plugins: {
			legend: {
				position: "top",
			},
		},
		scales: {
			x: {
				type: "category",
			},
			y: {
				beginAtZero: true,
			},
		},
	};

	return (
		<div className="rounded-2xl border bg-white p-6 shadow-sm">
			<div className="flex items-center justify-between gap-3 mb-4">
				<h4 className="font-medium">{title}</h4>
				<select className="rounded-xl border px-3 py-1 text-sm bg-white" value={selectedKey} onChange={(e) => onChangeKey(e.target.value as BarSeriesKey)}>
					{BAR_KEYS.map((key) => (
						<option key={key} value={key}>
							{CHART_SERIES_LABELS[key]}
						</option>
					))}
				</select>
			</div>

			<div className="h-64">
				<Chart type="bar" data={data} options={options} />
			</div>
		</div>
	);
}

export default function Analysis() {
	const [loading, setLoading] = React.useState(false);
	const [error, setError] = React.useState<string | null>(null);
	const [data, setData] = React.useState<any>(null);
	const [selected, setSelected] = React.useState<string | null>(null);
	const [status, setStatus] = React.useState<number | null>(null);
	const [durationMs, setDurationMs] = React.useState<number | null>(null);

	const [barSeriesKey, setBarSeriesKey] = React.useState<BarSeriesKey>("fred");
	const [selectedModel, setSelectedModel] = React.useState<string>("baseline");

	async function callEndpoint(key: string) {
		setSelected(key);
		setLoading(true);
		setError(null);
		setStatus(null);
		setDurationMs(null);
		setData(null);

		const url = `${API_BASE}/${key}/`;
		try {
			const t0 = performance.now();
			const res = await fetch(url);
			const t1 = performance.now();
			setDurationMs(Math.round(t1 - t0));
			setStatus(res.status);

			if (!res.ok) throw new Error(`HTTP ${res.status}`);

			const ct = res.headers.get("content-type") || "";
			const body = ct.includes("application/json") ? await res.json() : await res.text();
			setData(body);
		} catch (e: any) {
			setError(e?.message || "Błąd");
		} finally {
			setLoading(false);
		}
	}

	function handleRunModel() {
		// TODO: dodać wywołanie modelu ML
	}

	return (
		<section className="grid gap-6">
			<h2 className="text-2xl font-bold">Analiza</h2>

			{/* Karta z wywołaniem API, deprecated */}
			<div className="rounded-2xl border bg-white p-6 shadow-sm">
				<div className="flex flex-wrap items-center gap-2">
					{ENDPOINTS.map((ep) => {
						const active = selected === ep.key;
						return (
							<button
								key={ep.key}
								onClick={() => callEndpoint(ep.key)}
								className={`px-4 py-2 rounded-xl border transition ${active ? "bg-slate-900 text-white border-slate-900" : "bg-white text-slate-900 hover:bg-slate-50 border-slate-200"} disabled:opacity-50`}
								disabled={loading && active}
								title={`${API_BASE}/${ep.key}/`}>
								{loading && active ? "Ładowanie…" : ep.label}
							</button>
						);
					})}
				</div>

				<div className="mt-3 text-sm text-slate-600 flex flex-wrap gap-3">
					{selected && (
						<>
							<span className="font-medium">
								Endpoint: <code className="bg-slate-100 px-1 rounded">{selected}</code>
							</span>
							{status !== null && <span>Status: {status}</span>}
							{durationMs !== null && <span>Czas: {durationMs} ms</span>}
						</>
					)}
					{error && <span className="text-red-600">{error}</span>}
				</div>

				{!loading && !error && data == null && <p className="text-sm text-slate-500 mt-4">Wybierz jeden z endpointów powyżej, aby pobrać i wyświetlić odpowiedź.</p>}

				{data != null && <ResponseView value={data} />}
			</div>

			{/* Karta z wykresami + model ML */}
			<div className="rounded-2xl border bg-white p-6 shadow-sm">
				<div className="flex flex-wrap items-center justify-between gap-3 mb-4">
					<h3 className="text-lg font-semibold">Wykresy</h3>

					<div className="flex items-center gap-3 text-sm">
						<div className="flex items-center gap-2">
							<span className="text-slate-600">Model ML:</span>
							<select className="rounded-xl border px-3 py-1 bg-white" value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
								{ML_MODELS.map((m) => (
									<option key={m.key} value={m.key}>
										{m.label}
									</option>
								))}
							</select>
						</div>

						<button type="button" onClick={handleRunModel} className="inline-flex items-center rounded-xl bg-slate-900 px-4 py-1.5 text-sm font-medium text-white hover:bg-slate-800 transition">
							Uruchom model
						</button>
					</div>
				</div>

				<div className="grid gap-6">
					<BinanceCandleChart />

					<BarChartCard title="Wykres 2" selectedKey={barSeriesKey} onChangeKey={setBarSeriesKey} />
				</div>
			</div>
		</section>
	);
}
