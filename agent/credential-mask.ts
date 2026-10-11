export function createCredentialMask(apiKey: string): (value: unknown) => unknown {
	if ([...apiKey].length < 8) return (value) => value;
	const escape = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
	const jsonKey = JSON.stringify(apiKey).slice(1, -1);
	const asciiKey = jsonKey.replace(/[^\x00-\x7f]/g,
		(ch) => `\\u${ch.charCodeAt(0).toString(16).padStart(4, "0")}`);
	const serialized = [...new Set([jsonKey, asciiKey])]
		.filter((key) => key !== apiKey).sort((a, b) => b.length - a.length).map(escape);
	const encoded = [...apiKey].map((ch) => {
		const bytes = [...Buffer.from(ch)].map((byte) => `%${byte.toString(16).padStart(2, "0")}`)
			.join("").replace(/[a-f]/g, (hex) => `[${hex}${hex.toUpperCase()}]`);
		return `(?:${escape(ch)}|${bytes})`;
	}).join("");
	const pattern = new RegExp([...serialized, encoded].join("|"), "g");
	const maskText = (text: string) => text.replace(pattern, "***");
	const mask = (value: unknown): unknown => {
		if (typeof value === "string") return maskText(value);
		if (Array.isArray(value)) return value.map(mask);
		if (value && typeof value === "object") {
			return Object.fromEntries(Object.entries(value).map(([key, item]) => [maskText(key), mask(item)]));
		}
		return value;
	};
	return mask;
}
