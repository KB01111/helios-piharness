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
	extension: Extension;
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
	 *
	 * The previously installed extension with the same name is uninstalled first and the
	 * replacement takes the same position in the install order (hook/section order matters,
	 * e.g. guard-shaping must stay ahead of capture-telemetry). Replacing a toolpack's
	 * extension (`toolpack:<name>`) keeps its enabled/disabled state; a disabled pack stays
	 * uninstalled. An unknown name is installed as a new core extension.
	 *
	 * Note: conversations created with an explicit extension selection keep the extension
	 * objects they were created with; the replacement applies to registry-default selections
	 * and to conversations created afterwards.
	 */
	replaceExtension(extension: Extension): void {
		const pack = Array.from(this.toolPacks.values()).find((managed) => managed.extension.name === extension.name);
		if (pack) {
			const previous = pack.extension;
			pack.extension = extension;
			if (pack.enabled) {
				this.swapInstalled(previous, extension);
			}
			return;
		}

		const previous = this.coreExtensions.get(extension.name) ?? this.registry.snapshot().extension(extension.name);
		this.coreExtensions.set(extension.name, extension);
		if (previous) {
			this.swapInstalled(previous, extension);
		} else {
			this.registry.install(extension);
		}
	}

	/** Uninstalls `previous`, installs `next`, and restores the original install order. */
	private swapInstalled(previous: Extension, next: Extension): void {
		const installed = this.registry.snapshot().installed();
		const index = installed.findIndex((ext) => ext.name === previous.name);
		const trailing = index >= 0 ? installed.slice(index + 1) : [];

		this.registry.uninstall(previous);
		this.registry.install(next);
		// `install` appends after an uninstall: move everything that used to follow behind it again.
		for (const ext of trailing) {
			this.registry.uninstall(ext);
			this.registry.install(ext);
		}
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
