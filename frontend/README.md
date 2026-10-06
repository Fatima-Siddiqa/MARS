# frontend/ (Next.js + Tailwind + Lightweight Charts). Owner: Aneeqa.

Initialise in this folder (keep this README):

```bash
npx create-next-app@latest . --typescript --tailwind --eslint --app --src-dir --use-npm
npm i lightweight-charts
```

Set `NEXT_PUBLIC_API_URL=http://localhost:8000`. Add a `Dockerfile`, then uncomment the
`frontend` service in `docker-compose.yml`.
