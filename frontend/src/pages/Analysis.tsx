import React from "react";

type Endpoint = { key: string; label: string };

const API_BASE = "http://localhost:8000/marketdata";

const ENDPOINTS: Endpoint[] = [
	{ key: "binance-test", label: "Binance" },
	{ key: "fred-test", label: "FRED" },
	{ key: "stooq-test", label: "Stooq" },
	{ key: "coinmetrics-test", label: "CoinMetrics" },
];

export default function Analysis() {
	const [loading, setLoading] = React.useState(false);
	const [error, setError] = React.useState<string | null>(null);
	const [data, setData] = React.useState<any>(null);
	const [selected, setSelected] = React.useState<string | null>(null);
	const [status, setStatus] = React.useState<number | null>(null);
	const [durationMs, setDurationMs] = React.useState<number | null>(null);

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

	function ResponseView({ value }: { value: any }) {
		if (value === null || value === undefined) return null;
		if (typeof value === "string") {
			return <pre className="mt-3 max-h-[480px] overflow-auto rounded-xl bg-slate-900 p-4 text-slate-100 text-sm">{value}</pre>;
		}
		return <pre className="mt-3 max-h-[480px] overflow-auto rounded-xl bg-slate-900 p-4 text-slate-100 text-sm">{JSON.stringify(value, null, 2)}</pre>;
	}

	return (
		<section className="grid gap-4">
			<h2 className="text-2xl font-bold">Analiza</h2>

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
		</section>
	);
}
