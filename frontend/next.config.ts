import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // Silence multi-lockfile warning when repo root also has a lockfile.
  turbopack: {
    root: path.join(__dirname),
  },
};

export default nextConfig;
