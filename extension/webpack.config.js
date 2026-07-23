/**
 * gyAI extension — Webpack 5 build.
 *
 * Compiles the three TypeScript entry points (blueprint §13) and copies the
 * static assets into `dist/`, which is the folder you load as an unpacked
 * extension (chrome://extensions -> Load unpacked -> select `dist`).
 *
 * Output layout (paths referenced by manifest.json / injected at runtime):
 *   dist/
 *   ├── manifest.json
 *   ├── background/service_worker.js      (webNavigation listener)
 *   ├── content/content_script.js         (injected via chrome.scripting)
 *   ├── content/content_script.css        (loaded via runtime.getURL — needs web_accessible_resources)
 *   ├── popup/popup.js  +  popup.html  +  popup.css
 *   └── icons/icon16|48|128.png
 */
const path = require("path");
const CopyWebpackPlugin = require("copy-webpack-plugin");

module.exports = (env, argv) => {
  const isProduction = argv.mode === "production";

  return {
    mode: isProduction ? "production" : "development",

    // MV3 service workers / content scripts run under a strict CSP that
    // forbids eval(), so we must NOT use any "eval-*" devtool. Separate .map
    // files are CSP-safe; drop source maps entirely for the production build.
    devtool: isProduction ? false : "cheap-module-source-map",

    // Entry keys include the sub-folder so output preserves the structure
    // that manifest.json and the runtime injection code expect.
    entry: {
      "background/service_worker": "./src/background/service_worker.ts",
      "content/content_script": "./src/content/content_script.ts",
      "popup/popup": "./src/popup/popup.ts"
    },

    output: {
      path: path.resolve(__dirname, "dist"),
      filename: "[name].js",
      clean: true
    },

    resolve: {
      extensions: [".ts", ".js"]
    },

    module: {
      rules: [
        {
          test: /\.ts$/,
          use: "ts-loader",
          exclude: /node_modules/
        }
      ]
    },

    plugins: [
      new CopyWebpackPlugin({
        patterns: [
          // manifest is authored here and copied verbatim.
          { from: "manifest.json", to: "manifest.json" },
          // Assets that aren't imported through JS. These must all exist —
          // a missing one should fail the build, not ship a broken package.
          { from: "public/icons", to: "icons" },
          { from: "src/popup/popup.html", to: "popup/popup.html" },
          { from: "src/popup/popup.css", to: "popup/popup.css" },
          { from: "src/content/content_script.css", to: "content/content_script.css" }
        ]
      })
    ],

    optimization: {
      minimize: isProduction,
      // Keep each entry a single self-contained file. A classic MV3 service
      // worker / content script can't auto-load split or runtime chunks, so
      // both are disabled to avoid "chunk not found" failures at runtime.
      splitChunks: false,
      runtimeChunk: false
    },

    // Bundle sizes for an extension are irrelevant to page-load budgets.
    performance: { hints: false }
  };
};
