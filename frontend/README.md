# DoCode React Dashboard

This Vite + React app provides a lightweight UI for the DoCode job API.

## Features

- List visible DoCode jobs from `GET /v1/jobs`.
- Create a new job with the same fields accepted by `POST /v1/jobs`.
- Watch job status and step output in real time through `GET /v1/jobs/{job_id}/events`.
- Store an optional bearer token in local browser storage for authenticated deployments.

## Local development

Run the API first:

```bash
python -m pip install -e ".[dev]"
uvicorn docode.main:app --reload --port 8110
```

Then run the frontend dev server:

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/v1` and `/health` to `http://localhost:8110`, so the app can call the API without extra CORS configuration.

## Docker development

From the DoCodeDev root, start both services together:

```bash
docker compose up --build
```

The frontend container listens on `http://localhost:5173` and proxies API calls to the backend container.

## Production build

```bash
cd frontend
npm install
npm run build
```

When `frontend/dist` exists, `docode.main` mounts it at `/` after the API routes. The Dockerfile builds the frontend and copies the generated `dist` directory into the Python runtime image.

Set `VITE_DOCODE_API_BASE_URL` only when the built frontend should call a different API origin. Leave it unset for the default same-origin deployment.
