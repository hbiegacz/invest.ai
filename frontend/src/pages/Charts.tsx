import React from "react";

const data = [12, 8, 18, 5, 14, 10, 16];

export default function Charts() {
	const max = Math.max(...data);
	return (
		<section>
			<h2 className="text-2xl font-bold mb-4">Wykresy</h2>
			<div className="rounded-2xl border bg-white p-6 shadow-sm">
				<div className="flex items-end gap-2 h-48">
					{data.map((v, i) => (
						<div key={i} className="flex-1">
							<div className="w-full rounded-t-xl bg-indigo-500" style={{ height: `${(v / max) * 100}%` }} title={`Wartość: ${v}`} />
							<div className="text-center text-xs mt-1 text-slate-500">{i + 1}</div>
						</div>
					))}
				</div>
			</div>
		</section>
	);
}
