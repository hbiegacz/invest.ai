import React from "react";
import { Link } from "react-router-dom";

export default function Landing() {
	return (
		<section className="grid gap-6">
			<div className="rounded-2xl bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 p-8 text-white">
				<h1 className="text-3xl md:text-4xl font-bold">Witaj w aplikacji starter</h1>
				<p className="mt-2 opacity-90">Uwaga: to tylko prototyp wywołujący API!</p>
				<div className="mt-6 flex gap-3">
					<Link to="/wykresy" className="bg-white/10 hover:bg-white/20 px-4 py-2 rounded-xl">
						Zobacz wykresy
					</Link>
					<Link to="/analiza" className="bg-white text-slate-900 hover:bg-slate-100 px-4 py-2 rounded-xl">
						Analiza / API
					</Link>
				</div>
			</div>

			<div className="grid md:grid-cols-3 gap-4">
				{["Prototyp", "API", "React"].map((t) => (
					<div key={t} className="rounded-2xl border bg-white p-5 shadow-sm">
						<h3 className="font-semibold">{t}</h3>
						<p className="text-sm text-slate-600 mt-1">Lorem ipsum dolor sit amet.</p>
					</div>
				))}
			</div>
		</section>
	);
}
