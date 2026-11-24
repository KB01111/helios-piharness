import { serve } from "@hono/node-server";
import { loadConfig } from "./config.js";
import { bootstrapHarness } from "./index.js";
import { createHttpServer } from "./transport/http-server.js";

async function main(): Promise<void> {
	const config = loadConfig();
	const runtime = await bootstrapHarness({ config });
	const app = createHttpServer(runtime.harness);

	const server = serve(
		{
			fetch: app.fetch,
			hostname: config.host,
			port: config.port,
		},
		(info) => {
			console.log(
				JSON.stringify({
					level: "info",
					event: "server_started",
					url: `http://${info.address}:${info.port}`,
					mainModel: `${config.mainModel.provider}:${config.mainModel.modelId}`,
					fastModel: `${config.fastModel.provider}:${config.fastModel.modelId}`,
					useMockModels: config.useMockModels,
					piDbPath: config.piDbPath,
					appDbPath: config.appDbPath,
				}),
			);
		},
	);

	let shuttingDown = false;
	const shutdown = async (signal: string): Promise<void> => {
		if (shuttingDown) return;
		shuttingDown = true;
		console.log(JSON.stringify({ level: "info", event: "shutdown_started", signal }));

		await new Promise<void>((resolve) => {
			server.close(() => resolve());
		});

		try {
			await runtime.close();
		} finally {
			console.log(JSON.stringify({ level: "info", event: "shutdown_complete", signal }));
			process.exit(0);
		}
	};

	process.once("SIGINT", () => {
		void shutdown("SIGINT");
	});
	process.once("SIGTERM", () => {
		void shutdown("SIGTERM");
	});
}

main().catch((error) => {
	console.error("Fatal error starting Helios Pi Harness:", error);
	process.exit(1);
});
