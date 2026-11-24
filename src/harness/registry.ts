import {
	createRegistry,
	defineExtension,
	type Extension,
	type Registry,
	type ToolRegistration,
} from "@earendil-works/pi-durable";
import type { ToolPackSummary } from "../contracts/harness-port.js";

export interface ToolPackDefinition {
	readonly name: string;
	readonly description: string;
	readonly tools: readonly ToolRegistration[];
}

interface ManagedPack {
	readonly definition: ToolPackDefinition;
	readonly extension: Extension;
	enabled: boolean;
}

/**
 * Dynamic Pi Durable Extension Registry & ToolPack manager.
 *
 * Isolates core extensions (prompt, guard, capture, sleep-cycle, skills) from
 * toggleable ToolPacks so packs can be hot-toggled at runtime and individual
 * extensions can be hot-replaced in place (Loop 3 single-module patching).
 */
export class HarnessRegistryManager {
	private readonly registry: Registry;
	private readonly coreExtensions = new Map<string, Extension>();
	private readonly toolPacks = new Map<string, ManagedPack>();

	constructor() {
		this.registry = createRegistry();
	}

	getRegistry(): Registry {
		return this.registry;
	}

	installCoreExtension(extension: Extension): void {
		this.coreExtensions.set(extension.name, extension);
		this.registry.install(extension);
	}

	/**
	 * Hot-replaces a single extension in place by name without restarting the Harness.
	 */
	replaceExtension(extension: Extension): void {
		if (this.coreExtensions.has(extension.name)) {
			this.coreExtensions.set(extension.name, extension);
		}
		this.registry.install(extension);
	}

	uninstallCoreExtension(name: string): boolean {
		const existing = this.coreExtensions.get(name);
		if (!existing) return false;
		this.coreExtensions.delete(name);
		this.registry.uninstall(existing);
		return true;
	}

	registerToolPack(pack: ToolPackDefinition, options?: { readonly enabled?: boolean }): void {
		const enabled = options?.enabled ?? true;
		const extension = defineExtension({
			name: `toolpack:${pack.name}`,
			namespace: {
				name: pack.name,
				description: pack.description,
			},
			tools: pack.tools,
		});

		const existing = this.toolPacks.get(pack.name);
		if (existing && existing.enabled) {
			this.registry.uninstall(existing.extension);
		}

		this.toolPacks.set(pack.name, {
			definition: pack,
			extension,
			enabled,
		});

		if (enabled) {
			this.registry.install(extension);
		}
	}

	setToolPackEnabled(packName: string, enabled: boolean): void {
		const managed = this.toolPacks.get(packName);
		if (!managed) {
			throw new Error(`Unknown tool pack "${packName}"`);
		}
		if (managed.enabled === enabled) return;
		managed.enabled = enabled;
		if (enabled) {
			this.registry.install(managed.extension);
		} else {
			this.registry.uninstall(managed.extension);
		}
	}

	listToolPacks(): ToolPackSummary[] {
		return Array.from(this.toolPacks.values()).map((managed) => ({
			name: managed.definition.name,
			enabled: managed.enabled,
			tools: managed.definition.tools.map((t) => t.name),
		}));
	}

	getEnabledPackNames(): string[] {
		return Array.from(this.toolPacks.values())
			.filter((m) => m.enabled)
			.map((m) => m.definition.name);
	}

	resolveConversationExtensionSelection(enabledToolPacks?: readonly string[]): Extension[] | undefined {
		if (!enabledToolPacks) return undefined;
		const wanted = new Set(enabledToolPacks);
		const selected: Extension[] = Array.from(this.coreExtensions.values());
		for (const managed of this.toolPacks.values()) {
			if (wanted.has(managed.definition.name) && managed.enabled) {
				selected.push(managed.extension);
			}
		}
		return selected;
	}
}
