FROM node:20-slim

WORKDIR /app/frontend

COPY DoCodeDev/frontend/package.json ./package.json
COPY DoCodeDev/frontend/package-lock.json ./package-lock.json

RUN npm ci

COPY DoCodeDev/frontend/tsconfig.json ./tsconfig.json
COPY DoCodeDev/frontend/tsconfig.node.json ./tsconfig.node.json
COPY DoCodeDev/frontend/vite.config.ts ./vite.config.ts
COPY DoCodeDev/frontend/index.html ./index.html
COPY DoCodeDev/frontend/src ./src

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]