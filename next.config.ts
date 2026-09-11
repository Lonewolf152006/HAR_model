import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: false,

  // Proxy backend API calls from the dev server to the standalone AstroFlow-AI
  // service so the frontend can talk to a single origin during development.
  // The proxy is opt-in: set API_PROXY_TARGET to enable it, otherwise the
  // local route handlers under app/api/ serve the requests directly.
  async rewrites() {
    const target = process.env.API_PROXY_TARGET;
    if (!target) return [];
    return [
      {
        source: "/api/v1/:path*",
        destination: `${target}/api/v1/:path*`,
      },
    ];
  },
};

export default nextConfig;