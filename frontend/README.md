# CityMind AI — frontend

React 19 + TypeScript + Vite dashboard for the CityMind traffic-control project
(Story, Dashboard, Replay, Experiments, Comparison, XAI pages).

## Develop

```bash
npm install
npm run dev        # http://localhost:5173 (needs the API on :8000)
npm run lint       # oxlint
npm test           # vitest (jsdom)
npm run build      # tsc -b && vite build
```

## Configuration

Copy `.env.example` to `.env.local`. `VITE_API_URL` and `VITE_WS_URL` are optional in
development (defaults target `http://127.0.0.1:8000`) but **strongly recommended for production
builds** (`vite build` warns if `VITE_API_URL` is unset).

Do not put secrets in `VITE_*` variables: Vite inlines them into the public bundle.
The backend's control endpoints are loopback-only by default.
