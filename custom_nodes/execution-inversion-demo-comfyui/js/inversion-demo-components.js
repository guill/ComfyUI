import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";
import { ComfyDialog, $el } from "/scripts/ui.js";
import {ComfyWidgets} from "../../scripts/widgets.js";

const fileInput = $el("input", {
	id: "component-file-input",
	type: "file",
	accept: ".json,image/png,.latent,.safetensors",
	style: {display: "none"},
	parent: document.body,
	onchange: async () => {
		app.handleFile(fileInput.files[0]);
			const reader = new FileReader();
			reader.onload = () => {
				app.loadGraphData(JSON.parse(reader.result)["workflow"]);
			};
			reader.readAsText(fileInput.files[0]);
	},
});

api.addEventListener("custom_output.print_debug_text", ({ detail }) => {
	const node = app.graph.getNodeById(detail.node);
	if (node) {
		if (node.widgets) {
			const pos = node.widgets.findIndex((w) => w.name === "print_debug_text");
			if (pos !== -1) {
				for (let i = pos; i < node.widgets.length; i++) {
					node.widgets[i].onRemove?.();
				}
				node.widgets.length = pos;
			}
		}

		let text = detail.value;
		const w = ComfyWidgets["STRING"](node, "print_debug_text", ["STRING", { multiline: true }], app).widget;
		w.inputEl.readOnly = true;
		w.inputEl.style.opacity = 0.6;
		w.value = text;

		node.onResize?.(node.size);
	}
});

app.registerExtension({
	name: "Comfy.InversionDemoComponents",

	async setup() {
		const menu = document.querySelector(".comfy-menu");
		const separator = document.createElement("hr");

		separator.style.margin = "20px 0";
		separator.style.width = "100%";
		menu.append(separator);

		const saveButton = document.createElement("button");
		saveButton.textContent = "Save Component";
		saveButton.onclick = async () => {
			let filename = "component.json";
			const p = await app.graphToPrompt();
			const json = JSON.stringify(p, null, 2); // convert the data to a JSON string
			const blob = new Blob([json], {type: "application/json"});
			const url = URL.createObjectURL(blob);
			const a = $el("a", {
				href: url,
				download: filename,
				style: {display: "none"},
				parent: document.body,
			});
			a.click();
			setTimeout(function () {
				a.remove();
				window.URL.revokeObjectURL(url);
			}, 0);
		};

		const loadButton = document.createElement("button");
		loadButton.textContent = "Load Component";
		loadButton.onclick = () => {
			fileInput.click();
		};
		menu.append(saveButton);
		menu.append(loadButton);
	},
});
