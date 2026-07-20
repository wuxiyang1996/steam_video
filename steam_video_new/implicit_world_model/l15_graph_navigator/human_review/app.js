const FIELD_OPTIONS = {
  review_decision: ["accept", "reject"],
  observation_validity: ["valid", "invalid", "inconclusive"],
  action_execution_validity: ["valid", "invalid", "inconclusive"],
  belief_delta_validity: ["valid", "invalid", "inconclusive"],
  relation_outcome: ["supports", "rejects", "inconclusive", "not_applicable"],
};
const FIELD_LABELS = {
  review_decision: "记录是否可接纳",
  observation_validity: "Observation validity",
  action_execution_validity: "Action execution validity",
  belief_delta_validity: "Belief delta validity",
  relation_outcome: "Relation outcome",
};

let packet;
let state;
let current = 0;
let filter = "all";

const el = id => document.getElementById(id);

async function init() {
  const response = await fetch("/api/packet", { cache: "no-store" });
  packet = await response.json();
  const saved = JSON.parse(localStorage.getItem(storageKey()) || "null");
  state = saved && saved.packet_id === packet.packet_id ? saved : freshState();
  el("annotator").value = state.annotator || "";
  bindGlobalEvents();
  renderList();
  renderItem(0);
  updateProgress();
}

function freshState() {
  return {
    packet_id: packet.packet_id,
    annotator: "",
    annotations: Object.fromEntries(packet.items.map(item => [item.item_id, emptyAnnotation()])),
    flagged: {},
  };
}

function emptyAnnotation() {
  return {
    review_decision: null,
    observation_validity: null,
    action_execution_validity: null,
    belief_delta_validity: null,
    relation_outcome: null,
    evidence_refs: [],
    rationale: "",
  };
}

function bindGlobalEvents() {
  el("annotator").addEventListener("input", event => {
    state.annotator = event.target.value;
    save();
  });
  document.querySelectorAll(".filter").forEach(button => button.addEventListener("click", () => {
    document.querySelectorAll(".filter").forEach(value => value.classList.remove("active"));
    button.classList.add("active");
    filter = button.dataset.filter;
    renderList();
  }));
  el("previous").addEventListener("click", () => move(-1));
  el("next").addEventListener("click", () => move(1));
  el("flagged").addEventListener("change", event => {
    state.flagged[packet.items[current].item_id] = event.target.checked;
    save(); renderList();
  });
  el("rationale").addEventListener("input", event => {
    annotation().rationale = event.target.value;
    save(); renderList(); updateProgress();
  });
  el("resetDraft").addEventListener("click", resetDraft);
  el("validate").addEventListener("click", validateReview);
  el("exportReview").addEventListener("click", exportReview);
}

function renderList() {
  const list = el("itemList");
  list.innerHTML = "";
  packet.items.forEach((item, index) => {
    const done = isComplete(state.annotations[item.item_id]);
    const flagged = state.flagged[item.item_id];
    if (filter === "incomplete" && done) return;
    if (filter === "flagged" && !flagged) return;
    const button = document.createElement("button");
    button.className = `item-button${index === current ? " active" : ""}`;
    button.innerHTML = `<span class="index">${String(index + 1).padStart(2, "0")}</span><span class="kind">${escapeHtml(item.action.action_type)}</span><span class="dot ${flagged ? "flagged" : done ? "done" : ""}"></span>`;
    button.addEventListener("click", () => renderItem(index));
    list.appendChild(button);
  });
}

function renderItem(index) {
  current = Math.max(0, Math.min(packet.items.length - 1, index));
  const item = packet.items[current];
  const answer = state.annotations[item.item_id];
  el("reviewPane").hidden = false;
  el("itemPosition").textContent = `ITEM ${current + 1} / ${packet.items.length}`;
  el("question").textContent = item.question;
  el("actionType").textContent = `${item.action.action_type}: ${item.action.source_id || "—"} → ${(item.action.target_ids || []).join(", ") || "—"}`;
  el("relation").textContent = item.action.relation || "not specified";
  el("belief").textContent = `${item.belief_before.answerability}; missing: ${(item.belief_before.missing_roles || []).join(", ") || "none"}`;
  el("execution").textContent = `${item.executed_result.status}; ${item.executed_result.observation_outcome}`;
  el("flagged").checked = Boolean(state.flagged[item.item_id]);
  el("rationale").value = answer.rationale || "";
  renderEvidence(item);
  renderCategorical(answer);
  renderCitations(item, answer);
  el("previous").disabled = current === 0;
  el("next").textContent = current === packet.items.length - 1 ? "保存" : "保存并下一条";
  renderList();
}

function renderEvidence(item) {
  el("evidenceCards").innerHTML = (item.visible_context.nodes || []).map(node => {
    const states = (node.states || []).map(value => `${value.attribute}=${value.value}`);
    const people = (node.participants || []).map(value => `${value.role}:${value.surface}`);
    return `<article class="evidence-card"><div class="node-id">${escapeHtml(node.node_id || "")}</div><div>${escapeHtml(node.description || "No description")}</div><div class="chips">${[...people, ...states].map(value => `<span class="chip">${escapeHtml(value)}</span>`).join("")}</div></article>`;
  }).join("");
}

function renderCategorical(answer) {
  const container = el("categoricalFields");
  container.innerHTML = "";
  Object.entries(FIELD_OPTIONS).forEach(([field, options]) => {
    const wrapper = document.createElement("fieldset");
    wrapper.innerHTML = `<legend>${FIELD_LABELS[field]}</legend><div class="choice-group"></div>`;
    const group = wrapper.querySelector("div");
    options.forEach(option => {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio"; input.name = field; input.value = option;
      input.checked = answer[field] === option;
      input.addEventListener("change", () => {
        annotation()[field] = option;
        save(); renderList(); updateProgress();
      });
      label.append(input, document.createTextNode(option));
      group.appendChild(label);
    });
    container.appendChild(wrapper);
  });
}

function renderCitations(item, answer) {
  const refs = visibleRefs(item);
  const container = el("citationChoices");
  container.innerHTML = "";
  refs.forEach(ref => {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox"; input.checked = answer.evidence_refs.includes(ref);
    input.addEventListener("change", () => {
      const values = new Set(annotation().evidence_refs);
      input.checked ? values.add(ref) : values.delete(ref);
      annotation().evidence_refs = [...values];
      save(); renderList(); updateProgress();
    });
    const span = document.createElement("span"); span.textContent = ref;
    label.append(input, span); container.appendChild(label);
  });
}

function visibleRefs(item) {
  const refs = new Set([...(item.executed_result.evidence_refs || []), ...(item.executed_result.real_observation_ids || [])]);
  (item.visible_context.nodes || []).forEach(node => {
    if (node.node_id) refs.add(node.node_id);
    (node.evidence_refs || []).forEach(value => refs.add(value));
  });
  return [...refs].filter(Boolean).sort();
}

function annotation() { return state.annotations[packet.items[current].item_id]; }
function isComplete(value) {
  return Object.keys(FIELD_OPTIONS).every(field => Boolean(value[field])) && value.evidence_refs.length > 0 && value.rationale.trim().length > 0;
}
function move(delta) { save(); renderItem(current + delta); window.scrollTo({ top: 0, behavior: "smooth" }); }
function save() {
  localStorage.setItem(storageKey(), JSON.stringify(state));
  el("saveState").textContent = `已自动保存 · ${new Date().toLocaleTimeString()}`;
}
function storageKey() { return `steam-transition-review:${packet.packet_id}`; }
function updateProgress() {
  const done = packet.items.filter(item => isComplete(state.annotations[item.item_id])).length;
  el("progressText").textContent = `${done} / ${packet.items.length} 已完成`;
  el("progressBar").style.width = `${100 * done / packet.items.length}%`;
}

function buildReview() {
  return {
    schema_version: "steam-transition-review-response/v0.1",
    packet_id: packet.packet_id,
    labels_source: "independent_human",
    annotator: state.annotator.trim(),
    protocol: "outcome_blinded_categorical_human_review",
    decisions: packet.items.map(item => ({ item_id: item.item_id, ...state.annotations[item.item_id] })),
  };
}

function localErrors() {
  const missing = packet.items.filter(item => !isComplete(state.annotations[item.item_id]));
  const errors = [];
  if (!state.annotator.trim()) errors.push("请填写审核者身份");
  if (missing.length) errors.push(`仍有 ${missing.length} 条未完成`);
  return errors;
}

async function validateReview() {
  const errors = localErrors();
  if (errors.length) return toast(errors.join("；"));
  const response = await fetch("/api/validate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(buildReview()) });
  const result = await response.json();
  toast(result.valid ? "校验通过：可导出并锁定为 human_locked" : `校验失败：${result.error}`);
}

function exportReview() {
  const errors = localErrors();
  if (errors.length) return toast(errors.join("；"));
  const blob = new Blob([JSON.stringify(buildReview(), null, 2) + "\n"], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${packet.packet_id}.independent_human_review.json`;
  link.click(); URL.revokeObjectURL(link.href);
  toast("人工审核 JSON 已导出");
}

function resetDraft() {
  if (!confirm("确定清除本 packet 的全部本地审核草稿？")) return;
  localStorage.removeItem(storageKey()); state = freshState(); el("annotator").value = "";
  renderItem(0); updateProgress(); toast("本地草稿已清除");
}

function toast(message) {
  const node = el("toast"); node.textContent = message; node.classList.add("show");
  setTimeout(() => node.classList.remove("show"), 3200);
}
function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

init().catch(error => toast(`载入失败：${error.message}`));
