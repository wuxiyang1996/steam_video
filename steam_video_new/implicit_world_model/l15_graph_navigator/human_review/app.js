const FIELD_OPTIONS = {
  observation_validity: ["valid", "invalid", "inconclusive"],
  action_execution_validity: ["valid", "invalid", "inconclusive"],
  relation_outcome: ["supports", "rejects", "inconclusive", "not_applicable"],
  belief_delta_validity: ["valid", "invalid", "inconclusive"],
  review_decision: ["accept", "reject"],
};
const I18N = {
  zh: {
    title: "Executed Transition 人工审核", loading: "载入中…", reviewer: "审核者",
    reviewerPlaceholder: "姓名或 reviewer ID", itemFilters: "条目筛选", reviewItems: "审核条目",
    filterAll: "全部", filterIncomplete: "未完成", filterFlagged: "待复核", flagForReview: "待复核",
    videoEvidence: "真实视频证据", publicEvidence: "公开证据", categoricalJudgment: "分类判断",
    citations: "引用证据（至少一项）", rationale: "理由",
    rationalePlaceholder: "只依据上方公开证据说明判断；不要给数字 reward、score、confidence、probability 或 utility。",
    previous: "上一条", save: "保存", saveNext: "保存并下一条", draftStored: "草稿自动保存在本浏览器",
    resetDraft: "清除本地草稿", validate: "完整性校验", exportReview: "导出人工审核 JSON",
    completed: "{done} / {total} 已完成", autoSaved: "已自动保存 · {time}", notSpecified: "未指定",
    beliefSummary: "{answerability}；缺失：{missing}", none: "无", visualUnavailable: "未提供",
    visualPlayable: "{available} / {total} 时间窗可播放", intervalUnknown: "时间窗未知",
    noVisual: "此条目没有绑定视频证据。请仅依据公开文本证据判断，并在理由中注明。",
    videoMissing: "视频不可用：{reason}", replayWindow: "重播此时间窗", noDescription: "无描述",
    videoPartial: "警告：请求时间窗超过视频实际时长；只能播放已有部分（{reason}）。",
    reviewerRequired: "请填写审核者身份", itemsIncomplete: "仍有 {count} 条未完成",
    validationPassed: "校验通过：可导出并锁定为 human_locked", validationFailed: "校验失败：{error}",
    exported: "人工审核 JSON 已导出", resetConfirm: "确定清除本 packet 的全部本地审核草稿？",
    resetDone: "本地草稿已清除", loadFailed: "载入失败：{error}", assetsUnavailable: "审核资源不可用",
    visualMismatch: "视觉索引与审核 packet 不匹配",
    field_review_decision: "记录是否可接纳", field_observation_validity: "观察是否有效",
    field_action_execution_validity: "Action 执行是否有效", field_belief_delta_validity: "Belief delta 是否有效",
    field_relation_outcome: "关系结论",
    option_accept: "接纳", option_reject: "拒绝", option_valid: "有效", option_invalid: "无效",
    option_inconclusive: "无法判断", option_supports: "支持", option_rejects: "反驳",
    option_not_applicable: "不适用", role_source: "来源", role_target: "目标",
    role_observation: "观察", role_context: "上下文",
    evidenceLanguageNote: "证据描述保留源语言，避免翻译改变审核语义。",
    action: "推理操作", relation: "关系", belief: "Belief 状态", execution: "执行结果",
    action_verify: "验证", action_inspect_state_change: "检查状态变化", action_search_counterevidence: "搜索反证",
    action_semantic: "语义检索", action_temporal_back: "向前追溯", action_temporal_forward: "向后查找",
    action_track_entity: "跟踪实体", questionTemplate: "审核从 {source} 开始的候选“{kind}”推理路径。",
    reasoning_ambiguity_abstention: "歧义保留", reasoning_blocked_path_recovery: "阻塞路径恢复",
    reasoning_counterevidence: "反证搜索", reasoning_delayed_two_hop: "延迟两跳",
    reasoning_identity: "身份一致性", reasoning_state: "状态变化",
    guideTitle: "标注任务与判断标准",
    guideGoal: "你的任务不是给 planner 打分或提供 reward，而是判断执行这一步 reasoning action 后，真实证据是否支持它声称的 observation、relation 和 belief change。",
    guideGroundTruthTitle: "Ground truth 在这里如何使用",
    guideGroundTruthAvailableTitle: "已有的 terminal QA ground truth",
    guideGroundTruthAvailable: "当前 9 个视频共有 58 条 Video-Holmes QA GT，包含 Question、Options、Answer 和 Explanation。它可用于判断完整 trajectory 最终是否答对，并产生非数字的 pairwise preference。",
    guideGroundTruthMissingTitle: "缺少的 transition ground truth",
    guideGroundTruthMissing: "QA GT 没有当前 node pair 的 identity、同属性 before/after state、action execution 或 belief delta 标签，因此不能自动填写本页五个字段。",
    guideGroundTruthReviewTitle: "本次人工审核建立什么",
    guideGroundTruthReview: "你的 blind video judgment 将建立独立的 transition-level GT；审核锁定后，才会与 terminal QA GT 联合分析。",
    guideGroundTruthLeakageTitle: "防止 label leakage：",
    guideGroundTruthLeakage: "审核期间不展示正确答案、GT explanation、stored verifier outcome 或 hidden target。最终答案正确也不能证明中间 relation 正确，因为模型可能猜对或使用 shortcut。",
    guideDecisionOrder: "推荐判断顺序", guideFiveFields: "五个分类字段",
    guideStepObservation: "视频是否展示描述内容？→ Observation validity",
    guideStepAction: "系统是否正确完成搜索或验证？→ Action execution validity",
    guideStepRelation: "证据支持、反驳还是无法决定关系？→ Relation outcome",
    guideStepBelief: "证据是否足以改变当前 belief？→ Belief delta validity",
    guideStepRecord: "整条记录是否适合作为训练数据？→ Record admissibility",
    guideRecordName: "Record admissibility", guideRecordDefinition: "Accept 表示整条记录证据充分且可用于训练；存在错误或关键证据不足时选 Reject。",
    guideObservationName: "Observation validity", guideObservationDefinition: "判断画面是否真的展示声称的人物、动作、物体或状态。",
    guideActionName: "Action execution validity", guideActionDefinition: "判断 reasoning operation 是否被正确执行；没有找到反证仍可能是一次有效搜索。",
    guideBeliefName: "Belief delta validity", guideBeliefDefinition: "判断证据是否足以补全、推翻、解决矛盾或合理地保持不确定。",
    guideRelationName: "Relation outcome", guideRelationDefinition: "Supports=明确支持；Rejects=明确反驳；Inconclusive=证据不足；Not applicable=没有具体待验证关系。",
    guideStrictRules: "严格关系规则", guideIdentityTitle: "Identity / same_entity",
    guideIdentityRule: "只有外貌、服装、时空连续或同镜头跟踪等身份连续性证据才能选 Supports；仅仅都是“man/woman”不够。",
    guideStateTitle: "State transition", guideStateRule: "必须是同一实体、同一属性、明确 before/after value、顺序明确且确实发生变化。",
    guideCounterTitle: "Counterevidence", guideCounterRule: "只有直接反驳 hypothesis 才选 Rejects；相关内容或空搜索通常是 Inconclusive，不能自动算 Supports。",
    guideImportant: "重要：", guideImportantText: "Relation outcome=Rejects 不等于拒绝记录。若系统正确找到了反证，整条记录仍可 Accept。证据不足时请选择 Inconclusive，不要猜测。",
    guideMetricsTitle: "完成后统计什么", guideMetricsText: "我们会分别报告 observation/action/belief validity、relation confusion matrix、record acceptance rate、隐藏重复条目一致性，以及 identity/state strict precision；不会把它们压成单一数字 reward。",
    guideMetricValidity: "Validity distribution：分别统计 Valid / Invalid / Inconclusive 的数量与比例，用于定位 evidence、executor 或 belief target 的问题。",
    guideMetricConfusion: "Relation confusion matrix：人工结论为行、现有 verifier 结论为列，观察 Supports / Rejects / Inconclusive 的具体混淆。",
    guideMetricAcceptance: "Record acceptance rate = Accept 条目数 / 53；表示可进入后续训练集的比例。",
    guideMetricDuplicates: "Duplicate consistency：隐藏重复条目的五个分类判断是否完全一致，用于检查 reviewer 稳定性。",
    guideMetricPrecision: "Identity/state strict precision = 系统 admitted 且人工确认 Supports 的关系数 / 系统 admitted 的关系数；identity 与 state 分开计算。",
    plainContextTitle: "用简单语言理解这一条", plainSourceTitle: "SOURCE：推理从哪里开始",
    plainTargetTitle: "TARGET：action 想检查什么", plainObservationTitle: "OBSERVATION：执行后实际返回什么",
    plainRoleOverlap: "同一个节点可以同时是 TARGET 和 OBSERVATION：这表示系统请求检查它，并且执行后确实返回了它。",
    technicalContextTitle: "查看 technical graph context",
    annotationOrderNote: "请按 1→5 顺序填写；前四项判断局部证据，最后一项决定整条记录是否接纳。",
    noSourceContext: "没有明确 source。", noTargetsContext: "没有指定 target；重点检查实际返回结果。",
    noObservationsContext: "没有返回 observation。", relationClaim: "当前待检查关系：{relation}。",
    purpose_verify: "Planner 正在验证 source 与 target 之间的关系。",
    purpose_inspect_state_change: "Planner 正在检查同一实体的状态是否发生变化。",
    purpose_search_counterevidence: "Planner 正在搜索能直接反驳当前 hypothesis 的证据。",
    purpose_semantic: "Planner 正在按语义寻找可能补全 missing evidence 的内容。",
    purpose_temporal_back: "Planner 正在 source 之前寻找相关事件。",
    purpose_temporal_forward: "Planner 正在 source 之后寻找相关事件。",
    purpose_track_entity: "Planner 正在检查 source 与 target 是否属于同一实体。",
    purpose_default: "Planner 正在执行一个 reasoning action 来获取更多证据。",
    core_verify: "核心问题：返回的视频是否足以支持、反驳或无法判断当前 relation？",
    core_inspect_state_change: "核心问题：是否能看到同一实体、同一属性的明确 before→after value 变化？",
    core_search_counterevidence: "核心问题：返回内容是否直接反驳 hypothesis，还是仅仅相关或为空？",
    core_semantic: "核心问题：返回内容是否真实、相关，并足以改变当前 belief？",
    core_temporal_back: "核心问题：返回事件是否确实早于 source，并为当前推理提供有效证据？",
    core_temporal_forward: "核心问题：返回事件是否确实晚于 source，并为当前推理提供有效证据？",
    core_track_entity: "核心问题：是否有足够的视觉连续性证明是同一实体，而不只是同类称谓？",
    core_default: "核心问题：实际返回证据是否与 action 目标一致，并足以支持 belief update？",
    help_observation_validity: "只看画面：实际返回内容是否与公开 description 相符？",
    help_action_execution_validity: "看操作：系统是否返回了适合这个 action 的相关证据？",
    help_relation_outcome: "看关系：证据明确支持、明确反驳、证据不足，还是不适用？",
    help_belief_delta_validity: "看更新：谨慎的 agent 看完这些证据后，是否应该产生该 belief change？",
    help_review_decision: "最后填写：整条记录是否足够可靠，可以进入训练数据？",
  },
  en: {
    title: "Executed Transition Human Review", loading: "Loading…", reviewer: "Reviewer",
    reviewerPlaceholder: "Name or reviewer ID", itemFilters: "Item filters", reviewItems: "Review items",
    filterAll: "All", filterIncomplete: "Incomplete", filterFlagged: "Flagged", flagForReview: "Flag for review",
    videoEvidence: "Grounded video evidence", publicEvidence: "Public evidence", categoricalJudgment: "Categorical judgment",
    citations: "Evidence citations (at least one)", rationale: "Rationale",
    rationalePlaceholder: "Explain using only the public evidence above. Do not provide numeric reward, score, confidence, probability, or utility.",
    previous: "Previous", save: "Save", saveNext: "Save and next", draftStored: "Draft is stored automatically in this browser",
    resetDraft: "Clear local draft", validate: "Validate completeness", exportReview: "Export human review JSON",
    completed: "{done} / {total} completed", autoSaved: "Autosaved · {time}", notSpecified: "not specified",
    beliefSummary: "{answerability}; missing: {missing}", none: "none", visualUnavailable: "Unavailable",
    visualPlayable: "{available} / {total} time windows playable", intervalUnknown: "Unknown time window",
    noVisual: "No video evidence is bound to this item. Judge from the public text evidence only and note this in the rationale.",
    videoMissing: "Video unavailable: {reason}", replayWindow: "Replay this window", noDescription: "No description",
    videoPartial: "Warning: the requested window exceeds the actual video duration; only the existing portion is playable ({reason}).",
    reviewerRequired: "Enter a reviewer identity", itemsIncomplete: "{count} items remain incomplete",
    validationPassed: "Validation passed: ready to export and lock as human_locked", validationFailed: "Validation failed: {error}",
    exported: "Human review JSON exported", resetConfirm: "Clear the entire local draft for this packet?",
    resetDone: "Local draft cleared", loadFailed: "Load failed: {error}", assetsUnavailable: "Review assets unavailable",
    visualMismatch: "Visual index does not match the review packet",
    field_review_decision: "Record admissibility", field_observation_validity: "Observation validity",
    field_action_execution_validity: "Action execution validity", field_belief_delta_validity: "Belief delta validity",
    field_relation_outcome: "Relation outcome",
    option_accept: "Accept", option_reject: "Reject", option_valid: "Valid", option_invalid: "Invalid",
    option_inconclusive: "Inconclusive", option_supports: "Supports", option_rejects: "Rejects",
    option_not_applicable: "Not applicable", role_source: "Source", role_target: "Target",
    role_observation: "Observation", role_context: "Context",
    evidenceLanguageNote: "Evidence descriptions remain in their source language to avoid changing the review semantics through translation.",
    action: "Action", relation: "Relation", belief: "Belief", execution: "Execution",
    action_verify: "Verify", action_inspect_state_change: "Inspect state change", action_search_counterevidence: "Search counterevidence",
    action_semantic: "Semantic retrieval", action_temporal_back: "Temporal back", action_temporal_forward: "Temporal forward",
    action_track_entity: "Track entity", questionTemplate: "Review the proposed {kind} reasoning path from {source}.",
    reasoning_ambiguity_abstention: "ambiguity abstention", reasoning_blocked_path_recovery: "blocked path recovery",
    reasoning_counterevidence: "counterevidence", reasoning_delayed_two_hop: "delayed two-hop",
    reasoning_identity: "identity", reasoning_state: "state",
    guideTitle: "Annotation task and decision criteria",
    guideGoal: "Your task is not to score the planner or provide a reward. Judge whether the real evidence returned after this reasoning action supports the claimed observation, relation, and belief change.",
    guideGroundTruthTitle: "How ground truth is used here",
    guideGroundTruthAvailableTitle: "Available terminal QA ground truth",
    guideGroundTruthAvailable: "The nine current videos have 58 Video-Holmes QA ground-truth records containing Question, Options, Answer, and Explanation. They can determine whether a complete trajectory reaches the correct answer and can produce non-numeric pairwise preferences.",
    guideGroundTruthMissingTitle: "Missing transition ground truth",
    guideGroundTruthMissing: "QA ground truth does not label identity for the current node pair, same-attribute before/after state, action execution, or belief delta, so it cannot automatically fill the five fields on this page.",
    guideGroundTruthReviewTitle: "What this human review establishes",
    guideGroundTruthReview: "Your blinded video judgment creates independent transition-level ground truth. It will be joined with terminal QA ground truth only after the review is locked.",
    guideGroundTruthLeakageTitle: "Preventing label leakage: ",
    guideGroundTruthLeakage: "Correct answers, GT explanations, stored verifier outcomes, and hidden targets are not shown during review. A correct final answer does not prove that an intermediate relation is correct because the model may guess or exploit a shortcut.",
    guideDecisionOrder: "Recommended decision order", guideFiveFields: "Five categorical fields",
    guideStepObservation: "Does the video show the described content? → Observation validity",
    guideStepAction: "Was the search or verification operation completed correctly? → Action execution validity",
    guideStepRelation: "Does the evidence support, reject, or fail to decide the relation? → Relation outcome",
    guideStepBelief: "Is the evidence sufficient to change the current belief? → Belief delta validity",
    guideStepRecord: "Is the complete record suitable as training data? → Record admissibility",
    guideRecordName: "Record admissibility", guideRecordDefinition: "Accept when the complete record is sufficiently grounded and usable for training; Reject when it contains an error or lacks critical evidence.",
    guideObservationName: "Observation validity", guideObservationDefinition: "Judge whether the video actually shows the claimed person, action, object, or state.",
    guideActionName: "Action execution validity", guideActionDefinition: "Judge whether the reasoning operation was executed correctly. A counterevidence search can be valid even when it finds no counterevidence.",
    guideBeliefName: "Belief delta validity", guideBeliefDefinition: "Judge whether the evidence warrants completing or rejecting a link, resolving a contradiction, or deliberately remaining uncertain.",
    guideRelationName: "Relation outcome", guideRelationDefinition: "Supports=clearly supported; Rejects=clearly contradicted; Inconclusive=insufficient evidence; Not applicable=no concrete relation was tested.",
    guideStrictRules: "Strict relation rules", guideIdentityTitle: "Identity / same_entity",
    guideIdentityRule: "Choose Supports only with identity continuity such as appearance, clothing, spatiotemporal continuity, or same-shot tracking. Merely sharing the label “man/woman” is insufficient.",
    guideStateTitle: "State transition", guideStateRule: "Require the same entity, the same attribute, explicit before and after values, clear temporal order, and an actual value change.",
    guideCounterTitle: "Counterevidence", guideCounterRule: "Choose Rejects only for direct contradiction. Related content or an empty search is usually Inconclusive and does not automatically support the hypothesis.",
    guideImportant: "Important: ", guideImportantText: "Relation outcome=Rejects does not mean rejecting the record. If the system correctly discovered counterevidence, the record may still be Accept. Choose Inconclusive instead of guessing when evidence is insufficient.",
    guideMetricsTitle: "What will be measured", guideMetricsText: "We report observation/action/belief validity, the relation confusion matrix, record acceptance rate, hidden-duplicate consistency, and identity/state strict precision separately. They are not collapsed into a single numeric reward.",
    guideMetricValidity: "Validity distribution: counts and rates of Valid / Invalid / Inconclusive are reported separately to locate evidence, executor, or belief-target failures.",
    guideMetricConfusion: "Relation confusion matrix: human outcomes form the rows and existing verifier outcomes form the columns, exposing exact Supports / Rejects / Inconclusive confusions.",
    guideMetricAcceptance: "Record acceptance rate = number of Accept items / 53; this is the share eligible for the later training set.",
    guideMetricDuplicates: "Duplicate consistency: whether all five categorical decisions exactly agree on hidden duplicate items, measuring reviewer stability.",
    guideMetricPrecision: "Identity/state strict precision = system-admitted relations confirmed as Supports by the human / all system-admitted relations, computed separately for identity and state.",
    plainContextTitle: "Understand this item in plain language", plainSourceTitle: "SOURCE: where reasoning starts",
    plainTargetTitle: "TARGET: what the action intends to inspect", plainObservationTitle: "OBSERVATION: what execution actually returned",
    plainRoleOverlap: "A node may be both TARGET and OBSERVATION: the system requested it and execution actually returned it.",
    technicalContextTitle: "Show technical graph context",
    annotationOrderNote: "Complete fields 1→5 in order. The first four judge local evidence; the last decides whether to admit the complete record.",
    noSourceContext: "No explicit source.", noTargetsContext: "No target was specified; focus on the returned result.",
    noObservationsContext: "No observation was returned.", relationClaim: "Relation under review: {relation}.",
    purpose_verify: "The planner is verifying a relation between the source and target.",
    purpose_inspect_state_change: "The planner is checking whether the same entity changed state.",
    purpose_search_counterevidence: "The planner is searching for evidence that directly contradicts the current hypothesis.",
    purpose_semantic: "The planner is using semantic retrieval to find content that may fill missing evidence.",
    purpose_temporal_back: "The planner is looking for a relevant event before the source.",
    purpose_temporal_forward: "The planner is looking for a relevant event after the source.",
    purpose_track_entity: "The planner is checking whether the source and target refer to the same entity.",
    purpose_default: "The planner is executing a reasoning action to obtain more evidence.",
    core_verify: "Core question: does the returned video support, reject, or fail to decide the current relation?",
    core_inspect_state_change: "Core question: is there an explicit before→after value change for the same attribute of the same entity?",
    core_search_counterevidence: "Core question: does the result directly contradict the hypothesis, or is it merely related or empty?",
    core_semantic: "Core question: is the returned content real, relevant, and sufficient to change the current belief?",
    core_temporal_back: "Core question: does the returned event really occur before the source and provide useful evidence?",
    core_temporal_forward: "Core question: does the returned event really occur after the source and provide useful evidence?",
    core_track_entity: "Core question: is there enough visual continuity to establish the same entity, beyond a shared generic label?",
    core_default: "Core question: does the returned evidence match the action objective and warrant the belief update?",
    help_observation_validity: "Judge the video: does the returned content match its public description?",
    help_action_execution_validity: "Judge the operation: did the system return relevant evidence appropriate for this action?",
    help_relation_outcome: "Judge the relation: clearly supported, clearly rejected, insufficient evidence, or not applicable?",
    help_belief_delta_validity: "Judge the update: should a cautious agent make this belief change after seeing the evidence?",
    help_review_decision: "Fill this last: is the complete record reliable enough to enter the training data?",
  },
};

let packet;
let visualByItem = new Map();
let state;
let current = 0;
let filter = "all";
const storedLocale = localStorage.getItem("steam-review-language");
const requestedLocale = new URLSearchParams(window.location.search).get("lang");
let locale = ["zh", "en"].includes(requestedLocale)
  ? requestedLocale
  : (["zh", "en"].includes(storedLocale) ? storedLocale : (navigator.language.toLowerCase().startsWith("zh") ? "zh" : "en"));

const el = id => document.getElementById(id);

function t(key, values = {}) {
  let text = I18N[locale][key] ?? I18N.en[key] ?? key;
  Object.entries(values).forEach(([name, value]) => {
    text = text.replaceAll(`{${name}}`, String(value));
  });
  return text;
}

function applyLocale() {
  document.documentElement.lang = locale === "zh" ? "zh-CN" : "en";
  document.title = t("title");
  document.querySelectorAll("[data-i18n]").forEach(node => { node.textContent = t(node.dataset.i18n); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach(node => { node.placeholder = t(node.dataset.i18nPlaceholder); });
  document.querySelectorAll("[data-i18n-aria-label]").forEach(node => { node.setAttribute("aria-label", t(node.dataset.i18nAriaLabel)); });
  document.querySelectorAll(".language-option").forEach(button => {
    const active = button.dataset.language === locale;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  if (packet && state) {
    renderItem(current);
    updateProgress();
  }
}

function actionLabel(actionType) {
  const key = `action_${actionType}`;
  const translated = t(key);
  return translated === key ? actionType : translated;
}

function questionLabel(question) {
  if (locale === "en") return question;
  const match = /^Review the proposed (.+) reasoning path from (.+)\.$/.exec(question || "");
  if (!match) return question;
  const reasoningKey = `reasoning_${match[1].replaceAll(" ", "_")}`;
  return t("questionTemplate", { kind: t(reasoningKey), source: match[2] });
}

async function init() {
  applyLocale();
  const [packetResponse, visualResponse] = await Promise.all([
    fetch("/api/packet", { cache: "no-store" }),
    fetch("/api/visual", { cache: "no-store" }),
  ]);
  if (!packetResponse.ok || !visualResponse.ok) throw new Error(t("assetsUnavailable"));
  packet = await packetResponse.json();
  const visualIndex = await visualResponse.json();
  if (visualIndex.packet_id !== packet.packet_id) throw new Error(t("visualMismatch"));
  visualByItem = new Map((visualIndex.items || []).map(item => [item.item_id, item]));
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
  document.querySelectorAll(".language-option").forEach(button => button.addEventListener("click", () => {
    locale = button.dataset.language;
    localStorage.setItem("steam-review-language", locale);
    const url = new URL(window.location.href);
    url.searchParams.set("lang", locale);
    window.history.replaceState({}, "", url);
    applyLocale();
  }));
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
    button.innerHTML = `<span class="index">${String(index + 1).padStart(2, "0")}</span><span class="kind">${escapeHtml(actionLabel(item.action.action_type))}</span><span class="dot ${flagged ? "flagged" : done ? "done" : ""}"></span>`;
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
  el("question").textContent = questionLabel(item.question);
  el("actionType").textContent = `${actionLabel(item.action.action_type)} [${item.action.action_type}]: ${item.action.source_id || "—"} → ${(item.action.target_ids || []).join(", ") || "—"}`;
  el("relation").textContent = item.action.relation || t("notSpecified");
  el("belief").textContent = t("beliefSummary", { answerability: item.belief_before.answerability, missing: (item.belief_before.missing_roles || []).join(", ") || t("none") });
  el("execution").textContent = `${item.executed_result.status}; ${item.executed_result.observation_outcome}`;
  el("flagged").checked = Boolean(state.flagged[item.item_id]);
  el("rationale").value = answer.rationale || "";
  renderPlainContext(item);
  renderVisualEvidence(item);
  renderEvidence(item);
  renderCategorical(answer);
  renderCitations(item, answer);
  el("previous").disabled = current === 0;
  el("next").textContent = current === packet.items.length - 1 ? t("save") : t("saveNext");
  renderList();
}

function renderPlainContext(item) {
  const actionType = item.action.action_type;
  const purposeKey = `purpose_${actionType}`;
  const coreKey = `core_${actionType}`;
  const purpose = t(purposeKey) === purposeKey ? t("purpose_default") : t(purposeKey);
  const core = t(coreKey) === coreKey ? t("core_default") : t(coreKey);
  const relationText = item.action.relation ? ` ${t("relationClaim", { relation: item.action.relation })}` : "";
  el("plainActionSummary").textContent = purpose + relationText;
  el("plainCoreQuestion").textContent = core;

  const nodes = new Map((item.visible_context.nodes || []).map(node => [node.node_id, node]));
  const source = nodes.get(item.action.source_id);
  el("plainSource").textContent = source ? contextDescription(source) : (item.action.source_id || t("noSourceContext"));
  renderContextNodes("plainTargets", item.action.target_ids || [], nodes, "noTargetsContext");
  renderContextNodes("plainObservations", item.executed_result.real_observation_ids || [], nodes, "noObservationsContext");
}

function renderContextNodes(elementId, ids, nodes, emptyKey) {
  const container = el(elementId);
  container.innerHTML = "";
  if (!ids.length) {
    container.textContent = t(emptyKey);
    return;
  }
  const list = document.createElement("ul");
  [...new Set(ids)].forEach(nodeId => {
    const item = document.createElement("li");
    item.textContent = nodes.has(nodeId) ? contextDescription(nodes.get(nodeId)) : nodeId;
    list.appendChild(item);
  });
  container.appendChild(list);
}

function contextDescription(node) {
  const text = String(node.description || node.node_id || t("noDescription"));
  return text.length > 240 ? `${text.slice(0, 237)}…` : text;
}

function renderVisualEvidence(item) {
  const visual = visualByItem.get(item.item_id);
  const assets = visual ? (visual.visual_evidence || []) : [];
  const container = el("visualEvidence");
  container.innerHTML = "";
  if (!assets.length) {
    container.innerHTML = `<div class="visual-missing">${escapeHtml(t("noVisual"))}</div>`;
    el("visualStatus").textContent = t("visualUnavailable");
    return;
  }
  const available = assets.filter(asset => asset.availability !== "missing").length;
  el("visualStatus").textContent = t("visualPlayable", { available, total: assets.length });
  assets.forEach(asset => {
    const card = document.createElement("article");
    card.className = "visual-card";
    const roles = (asset.roles || []).map(role => `<span class="role role-${escapeHtml(role)}">${escapeHtml(t(`role_${role}`))}</span>`).join("");
    const interval = asset.start_s == null || asset.end_s == null ? t("intervalUnknown") : `${formatTime(asset.start_s)} – ${formatTime(asset.end_s)}`;
    card.innerHTML = `<div class="visual-meta"><div class="roles">${roles}</div><span>${escapeHtml(interval)}</span></div><div class="node-id">${escapeHtml(asset.node_id || "")}</div>`;
    if (asset.availability === "missing") {
      const missing = document.createElement("div");
      missing.className = "visual-missing";
      missing.textContent = t("videoMissing", { reason: asset.missing_reason || "unknown" });
      card.appendChild(missing);
    } else {
      if (asset.availability === "partial") {
        const warning = document.createElement("div");
        warning.className = "visual-missing";
        warning.textContent = t("videoPartial", { reason: asset.missing_reason || "unknown" });
        card.appendChild(warning);
      }
      const video = document.createElement("video");
      video.controls = true;
      video.preload = "metadata";
      video.src = asset.media_url;
      video.setAttribute("playsinline", "");
      const seekStart = () => {
        if (Number.isFinite(asset.start_s) && Math.abs(video.currentTime - asset.start_s) > 0.35) video.currentTime = asset.start_s;
      };
      video.addEventListener("loadedmetadata", seekStart, { once: true });
      video.addEventListener("play", () => {
        if (video.currentTime < asset.start_s || video.currentTime >= asset.end_s) seekStart();
      });
      video.addEventListener("timeupdate", () => {
        if (!video.paused && video.currentTime >= asset.end_s) video.pause();
      });
      const replay = document.createElement("button");
      replay.type = "button";
      replay.className = "secondary replay-window";
      replay.textContent = t("replayWindow");
      replay.addEventListener("click", async () => {
        video.currentTime = asset.start_s;
        try { await video.play(); } catch (_) { /* browser retains controls */ }
      });
      card.append(video, replay);
    }
    container.appendChild(card);
  });
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${(seconds - minutes * 60).toFixed(1).padStart(4, "0")}`;
}

function renderEvidence(item) {
  el("evidenceCards").innerHTML = (item.visible_context.nodes || []).map(node => {
    const states = (node.states || []).map(value => `${value.attribute}=${value.value}`);
    const people = (node.participants || []).map(value => `${value.role}:${value.surface}`);
    return `<article class="evidence-card"><div class="node-id">${escapeHtml(node.node_id || "")}</div><div>${escapeHtml(node.description || t("noDescription"))}</div><div class="chips">${[...people, ...states].map(value => `<span class="chip">${escapeHtml(value)}</span>`).join("")}</div></article>`;
  }).join("");
}

function renderCategorical(answer) {
  const container = el("categoricalFields");
  container.innerHTML = "";
  Object.entries(FIELD_OPTIONS).forEach(([field, options], fieldIndex) => {
    const wrapper = document.createElement("fieldset");
    wrapper.innerHTML = `<legend><span class="field-step">${fieldIndex + 1}</span>${escapeHtml(t(`field_${field}`))}</legend><p class="field-help">${escapeHtml(t(`help_${field}`))}</p><div class="choice-group"></div>`;
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
      label.append(input, document.createTextNode(t(`option_${option}`)));
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
  el("saveState").textContent = t("autoSaved", { time: new Date().toLocaleTimeString(locale === "zh" ? "zh-CN" : "en-US") });
}
function storageKey() { return `steam-transition-review:${packet.packet_id}`; }
function updateProgress() {
  const done = packet.items.filter(item => isComplete(state.annotations[item.item_id])).length;
  el("progressText").textContent = t("completed", { done, total: packet.items.length });
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
  if (!state.annotator.trim()) errors.push(t("reviewerRequired"));
  if (missing.length) errors.push(t("itemsIncomplete", { count: missing.length }));
  return errors;
}

async function validateReview() {
  const errors = localErrors();
  if (errors.length) return toast(errors.join(locale === "zh" ? "；" : "; "));
  const response = await fetch("/api/validate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(buildReview()) });
  const result = await response.json();
  toast(result.valid ? t("validationPassed") : t("validationFailed", { error: result.error }));
}

function exportReview() {
  const errors = localErrors();
  if (errors.length) return toast(errors.join(locale === "zh" ? "；" : "; "));
  const blob = new Blob([JSON.stringify(buildReview(), null, 2) + "\n"], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${packet.packet_id}.independent_human_review.json`;
  link.click(); URL.revokeObjectURL(link.href);
  toast(t("exported"));
}

function resetDraft() {
  if (!confirm(t("resetConfirm"))) return;
  localStorage.removeItem(storageKey()); state = freshState(); el("annotator").value = "";
  renderItem(0); updateProgress(); toast(t("resetDone"));
}

function toast(message) {
  const node = el("toast"); node.textContent = message; node.classList.add("show");
  setTimeout(() => node.classList.remove("show"), 3200);
}
function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

init().catch(error => toast(t("loadFailed", { error: error.message })));
