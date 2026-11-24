import { int, obj, opt, str } from "@earendil-works/pi-schema";
import type { ArtifactStore } from "../contracts/app-capabilities.js";
import type { ToolPackDefinition } from "../harness/registry.js";
import { createTenantScopedTool, defineToolPack } from "./tool-pack.js";

const ArtifactReadParams = obj({
	artifactId: str({ description: "Artifact ID returned when a large tool output was offloaded." }),
	offset: opt(int({ minimum: 0, description: "Character offset to start reading from (default 0)." })),
	limit: opt(int({ minimum: 1, maximum: 8192, description: "Maximum characters to read in one chunk (default 4000)." })),
});

export function createArtifactToolPack(artifacts: ArtifactStore): ToolPackDefinition {
	const artifactRead = createTenantScopedTool({
		name: "artifact_read",
		description: "Read a slice of an offloaded large tool output artifact by artifactId. Safe to replay.",
		parameters: ArtifactReadParams,
		replay: "safe",
		execute: async (args, tenant) => {
			const result = await artifacts.get(tenant.tenantId, tenant.userId, args.artifactId, {
				offset: args.offset ?? 0,
				limit: args.limit ?? 4000,
			});
			if (!result) {
				return {
					isError: true,
					output: [{ type: "text", text: `Artifact "${args.artifactId}" not found.` }],
				};
			}
			const header = `[artifact ${result.metadata.id} | offset=${result.offset}..${result.offset + result.chunk.length} | totalBytes=${result.totalBytes} | more=${result.truncated}]`;
			return {
				output: [{ type: "text", text: `${header}\n${result.chunk}` }],
				details: {
					artifactId: result.metadata.id,
					offset: result.offset,
					returnedLength: result.chunk.length,
					truncated: result.truncated,
				},
			};
		},
	});

	return defineToolPack({
		name: "artifacts",
		description: "Paginated retrieval of large tool outputs offloaded to artifact storage.",
		tools: [artifactRead],
	});
}
