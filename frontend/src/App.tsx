import React from "react";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import Analysis from "./pages/Analysis";

const linkClass = ({ isActive }: { isActive: boolean }) => `px-3 py-2 rounded-xl transition hover:bg-slate-100 ${isActive ? "bg-slate-200 font-medium" : ""}`;

export default function App() {
	return (
		<BrowserRouter>
			<div className="min-h-screen bg-slate-50 text-slate-900">
				<header className="sticky top-0 z-10 bg-white/80 backdrop-blur border-b">
					<nav className="max-w-5xl mx-auto flex gap-2 p-4">
						<NavLink to="/" className={linkClass} end>
							Analiza
						</NavLink>
					</nav>
				</header>

				<main className="max-w-5xl mx-auto p-6">
					<Routes>
						<Route path="/" element={<Analysis />} />
					</Routes>
				</main>

				<footer className="text-center text-sm text-slate-500 py-6">© {new Date().getFullYear()} app</footer>
			</div>
		</BrowserRouter>
	);
}
