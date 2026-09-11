import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: false,

  // Proxy backend API calls from the dev server to the standalone AstroFlow-AI
  // service so the frontend can talk to a single origin during development.
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: `${process.env.API_PROXY_TARGET ?? "http://localhost:8080"}/api/v1/:path*`,
      },
    ];
  },
};

export default nextConfig;