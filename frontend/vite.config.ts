import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '');
  const apiProxyTarget = env.DOCODE_API_PROXY_TARGET || env.VITE_DOCODE_API_PROXY_TARGET || 'http://127.0.0.1:8111';

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        '/v1': apiProxyTarget,
        '/health': apiProxyTarget
      }
    }
  };
});
