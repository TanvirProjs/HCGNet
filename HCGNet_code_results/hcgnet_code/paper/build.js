// Builds the IEEE A4 conference paper (.docx). Every number comes from results/nums.json.
const fs = require("fs");
const L = require("./lib.js");
const { algoBox, d, PT, FONT, runs, body, bodyNoIndent, h1, h2, eq, bullet, tableHead, table, figure, figCaption, ref } = L;
const { Document, Packer, Paragraph, TextRun, AlignmentType, SectionType, Footer } = d;

const imgH = (p, w) => { const b = fs.readFileSync(p); return w * b.readUInt32BE(20) / b.readUInt32BE(16); };
const FIG1_H = imgH(__dirname + "/../results/fig_arch.png", 236);
const FIG2_H = imgH(__dirname + "/../results/fig_size.png", 236);
const FIG3_H = imgH(__dirname + "/../results/fig_parity.png", 236);
const N = JSON.parse(fs.readFileSync(__dirname + "/../results/nums.json", "utf8"));
const REFS = require("./refs.js");
const ORDER = [];
const citeNum = (k) => {
  if (!(k in REFS)) throw new Error("unknown reference: " + k);
  if (!ORDER.includes(k)) ORDER.push(k);
  return ORDER.indexOf(k) + 1;
};
const fmtCite = (keys) => {
  const n = keys.map(citeNum).sort((a, b) => a - b);
  const parts = []; let i = 0;
  while (i < n.length) {
    let j = i; while (j + 1 < n.length && n[j + 1] === n[j] + 1) j++;
    if (j - i >= 2) parts.push(`[${n[i]}]\u2013[${n[j]}]`); else for (let k = i; k <= j; k++) parts.push(`[${n[k]}]`);
    i = j + 1;
  }
  return parts.join(", ");
};
const S = (t) => t.replace(/\[@([a-z0-9_,@ ]+)\]/g, (_, ks) => fmtCite(ks.split(",").map((x) => x.trim().replace(/^@/, ""))))
  .replace(/\{\{([a-zA-Z0-9_]+)\}\}/g, (_, k) => {
    if (!(k in N)) { if (process.env.STRICT) throw new Error("missing number: " + k); console.warn("missing", k); return "??"; }
    return String(N[k]);
  });
const P = (t) => body(S(t));
const E = (t, n) => eq(S(t), n);
const BL = (t) => bullet(S(t));

// ------------------------------------------------------------------ front matter
const TITLE = "Learning to Predict Transpilation Cost: A Graph Neural Network Surrogate for Post-Routing Depth, Two-Qubit Gate Count and Estimated Fidelity";
const titleBlock = [new Paragraph({
  children: [new TextRun({ text: TITLE, font: FONT, size: 48 })],
  alignment: AlignmentType.CENTER, spacing: { after: 6 * PT },
})];
const authorBlock = [
  new Paragraph({ children: [new TextRun({ text: "Gousia Thahniyath", font: FONT, size: 22 })], alignment: AlignmentType.CENTER, spacing: { before: 18 * PT, after: 2 * PT } }),
  new Paragraph({ children: [new TextRun({ text: "Department of Computer Science and Engineering", font: FONT, size: 18, italics: true })], alignment: AlignmentType.CENTER }),
  new Paragraph({ children: [new TextRun({ text: "Dayananda Sagar University", font: FONT, size: 18, italics: true })], alignment: AlignmentType.CENTER }),
  new Paragraph({ children: [new TextRun({ text: "Bengaluru, India", font: FONT, size: 18 })], alignment: AlignmentType.CENTER }),
  new Paragraph({ children: [new TextRun({ text: "gousiaaipro@gmail.com", font: FONT, size: 18 })], alignment: AlignmentType.CENTER, spacing: { after: 12 * PT } }),
];

const abs = (lead, text, kw = false) => new Paragraph({
  children: [new TextRun({ text: lead, font: FONT, size: 18, bold: true, italics: true }),
    ...runs(S(text), { size: 18, bold: true, italics: kw ? true : undefined })],
  alignment: AlignmentType.JUSTIFIED, indent: { firstLine: Math.round((kw ? 13.7 : 13.6) * PT) },
  spacing: { after: (kw ? 6 : 10) * PT },
});

const POS = [
  ["Work", "Input", "Predicts", "Model"],
  ["QuEst [@quest]", "compiled", "fidelity (PST)", "graph transf."],
  ["Saravanan [@saravanan23]", "compiled", "output fidelity", "ML, GNN"],
  ["MQT Pred. [@mqtpred]", "logical", "device, options", "classifier"],
  ["MLQM [@mlqm]", "logical", "optimal depth, SWAPs", "trees"],
  ["Tudisco [@tudisco25qce]", "logical", "best device", "GNN classif."],
  ["Tudisco [@tudisco26qsw]", "logical", "ESP per QPU", "GNN regr."],
  ["Singh [@singh26]", "logical + device", "added gates, wPST", "two-step ML"],
  ["{b:This work}", "logical + OL", "depth, N{sub:2q}, ESP", "GNN + GBDT"],
];
const C = [];
C.push(abs("Abstract\u2014", "Routing a logical circuit onto a device with restricted connectivity inserts SWAPs whose depth, two-qubit-gate and fidelity cost is known only after transpilation, yet design-space exploration, device selection and scheduling must rank many candidates before compiling them. We formulate transpilation-cost prediction as multi-target regression from the device-agnostic logical circuit and present HCG-Net, a heterogeneous graph neural network over consolidated two-qubit blocks and logical qubits that predicts post-routing depth, two-qubit gate count and estimated success probability (ESP) as log-overheads above a structural prior. We release a dataset of {{sherbrooke_count}} circuits from nine families transpiled with Qiskit at optimisation levels 1\u20133 on a calibrated 127-qubit Eagle model, plus {{torino_count}} circuits on a 133-qubit Heron model. Because SABRE-based compilation is stochastic, we first measure the seed-to-seed noise floor ({{nf_sherbrooke_depth}}%, {{nf_sherbrooke_twoq}}% and {{nf_sherbrooke_esp}}% for depth, two-qubit count and \u2212ln ESP). Over {{n_seeds}} random splits HCG-Net attains {{hcgnet_id_depth}}%, {{hcgnet_id_twoq}}% and {{hcgnet_id_esp}}% mean absolute percentage error, and its ensemble with a descriptor-based gradient-boosted model reaches {{hcg_ens_id_depth}}%, {{hcg_ens_id_twoq}}% and {{hcg_ens_id_esp}}%, within {{ratio_hcg_ens_depth}}\u00d7, {{ratio_hcg_ens_twoq}}\u00d7 and {{ratio_hcg_ens_esp}}\u00d7 of the floor, while answering about {{speedupb}}\u00d7 faster than transpilation in batch mode. On wider, unseen circuits errors rise by 1.6\u20132.1\u00d7; the structural prior most limits this degradation. Code and data are released."));
C.push(abs("Keywords\u2014", "quantum compilation, qubit routing, graph neural networks, surrogate modelling, estimated success probability, NISQ", true));

// ------------------------------------------------------------------ I. Introduction
C.push(h1(1, "Introduction"));
C.push(P("Noisy intermediate-scale quantum (NISQ) processors execute only circuits that respect their native gate set and qubit connectivity [@preskill]. Superconducting devices from IBM use a heavy-hexagon lattice in which every qubit has at most three neighbours [@heavyhex], so a logical two-qubit gate between non-adjacent qubits must be preceded by SWAP operations. Choosing the initial layout and the SWAP sequence is the qubit mapping and routing problem; production compilers solve it heuristically, most prominently with SABRE [@sabre] and its LightSABRE successor in Qiskit [@lightsabre, @qiskit], or with the routing passes of t|ket⟩ [@tket], and a 2025 survey reviews solver-based, heuristic and learning-based mappers for superconducting, trapped-ion and neutral-atom devices [@survey25]. The inserted SWAPs increase circuit depth and the two-qubit gate count N{sub:2q}, and therefore reduce the probability that the circuit runs without error."));
C.push(P("These post-routing costs are needed long before a circuit is compiled for execution. Variational-algorithm designers compare ansätze and gate orderings, device-selection tools compare back-ends [@mqtpred], and multi-QPU schedulers in hybrid HPC–quantum systems must estimate the fidelity of many queued circuits on each device [@fulginiti26, @tudisco26qsw]. Compiling each candidate is accurate but slow (on average {{t_transpile_ms}} ms here, up to 4.7 s for 28-qubit circuits), and SABRE is randomised, so a reliable estimate needs several trials; a learned surrogate turns this into a millisecond query."));
C.push(P("Learned predictors in quantum compilation have multiplied since 2025: graph neural networks (GNNs) select the best device [@tudisco25qce] or predict the estimated success probability (ESP) of an uncompiled circuit on several QPUs [@tudisco26qsw], earlier models predict the fidelity of compiled circuits [@quest], and two-step models estimate the gates a compiler adds before a figure of merit [@singh26]. Tree regressors have been used to predict the solver-optimal depth and SWAP count that prune an exact mapper [@mlqm]. None of these studies jointly regresses the depth, N{sub:2q} and ESP produced by a production transpiler, quantifies how much of the target is irreducible because the compiler itself is stochastic, or tests extrapolation to wider circuits and unseen circuit families. This paper addresses these gaps with four contributions."));
C.push(BL("{i:Problem and dataset.} We cast transpilation-cost prediction as multi-target regression and release {{sherbrooke_count}} + {{torino_count}} labelled circuits from nine families on two calibrated IBM device models, with five-seed repeats on a subset."));
C.push(BL("{i:HCG-Net.} A heterogeneous circuit graph of consolidated two-qubit blocks and logical qubits, a size-intensive readout and a structural prior that turns the targets into log-overheads."));
C.push(BL("{i:Noise-floor-aware protocol.} We measure the compiler’s irreducible seed-to-seed error and evaluate random splits, unseen widths and families, and a second device, with paired significance tests."));
C.push(BL("{i:Findings.} The best surrogate is within {{ratio_hcg_ens_depth}}–{{ratio_hcg_ens_esp}}× of the noise floor in distribution; we also report where it fails, including a strong tree-ensemble baseline that HCG-Net alone does not beat in distribution."));

// ------------------------------------------------------------------ II. Related work
C.push(h1(2, "Related Work"));
C.push(h2("A", "Mapping, Routing and Learned Routers"));
C.push(P("Zulehner {i:et al.} [@zulehner] gave an A*-based mapper for IBM QX devices and Li {i:et al.} proposed SABRE [@sabre], which remains the Qiskit default after the LightSABRE re-engineering [@lightsabre]. Noise-adaptive mappers steer layouts toward reliable qubits and couplers [@murali], using the ESP of Nishio {i:et al.} [@nishio] as a figure of merit. Learned routers combine reinforcement learning with GNNs [@saravanan24], power the RL-based routing and synthesis passes of the Qiskit transpiler [@kremer], and in 2025–2026 target fidelity directly, through a Gaussian-process fidelity surrogate inside an RL router [@fiddle] or calibration-aware graph RL on Heron devices [@tomar26]. Bhattacharjee {i:et al.} [@bhattacharjee25] showed how post-compilation SWAP overhead, depth and N{sub:2q} scale with entanglement structure and topology. All of these produce or analyse compiled circuits; we predict a compiler’s output without running it."));
C.push(h2("B", "Learned Fidelity and Cost Prediction"));
C.push(P("QuEst [@quest] regresses the fidelity of compiled circuits annotated with calibration data and Saravanan and Saeed [@saravanan23] compare classical regressors with GNNs for compiled-circuit reliability, and GNNs have since been used to predict execution success [@tudisco26rc] and circuit output expectation values [@liu26]. Hopf {i:et al.} [@hopf25] and Singh {i:et al.} [@singh26] learn improved figures of merit for compilation; the latter predicts the gates a compiler will add before predicting a weighted success probability. Closest to our work, Tudisco {i:et al.} [@tudisco26qsw] predict ESP on three QPUs from uncompiled circuits with a GNN; their labels are the best ESP over optimisation levels, circuits have at most 15 qubits, and neither compiler stochasticity nor size extrapolation is studied. Fulginiti {i:et al.} [@fulginiti26] use such predictions to schedule circuits across QPUs."));
C.push(h2("C", "Learned Compilation Decisions"));
C.push(P("MQT Predictor classifies the best device and compilation options [@qsw23, @mqtpred], Tudisco {i:et al.} [@tudisco25qce] classify the best device with a GNN, and Malkowski {i:et al.} [@malkowski26] learn which Qiskit pass combinations reduce N{sub:2q}. MLQM [@mlqm] predicts solver-optimal depth and SWAP count to prune an exact mapper, and QXX-MLP [@qxx] approximates the depth of its own layout heuristic. These rank discrete choices or model an exact solver; we regress continuous costs of a stochastic production heuristic."));
C.push(h2("D", "Research Gaps and Design Response"));
C.push(...tableHead("I", "Positioning Against Learned Predictors"));
C.push(table(POS.map((r) => r.map((c) => S(c))), [70, 50, 75, 48]));
C.push(P("Table I and the survey above expose four gaps that shape our method. G1: no model jointly regresses depth, N{sub:2q} and ESP of a production transpiler; we train one multi-target network (Section III-D). G2: labels are treated as deterministic, although SABRE is randomised; we measure the seed-noise floor and train for the seed-averaged cost (Sections III-A, IV-D). G3: evaluations stay within the training width range and circuit families; we add a structural prior and size-intensive readout for extrapolation (Sections III-C, III-D) and test unseen widths and families (Sections V-B, V-D). G4: GNNs are often compared with weak baselines; we compare against tuned tree ensembles on the same descriptors and prior, with paired significance tests (Section V-A)."));

// ------------------------------------------------------------------ III. Method
C.push(h1(3, "Problem Formulation and Method"));
C.push(h2("A", "Transpilation-Cost Regression"));
C.push(P("Let {i:C} be a logical circuit on {i:n} qubits, {i:D} a device with coupling graph {i:G}{isub:D} and calibrated error rates, and {i:T}{isub:o,s} a transpiler with options {i:o} (here the optimisation level) and random seed {i:s}. The compiled circuit {i:C}\u2032 = {i:T}{isub:o,s}({i:C}, {i:D}) has depth {i:d}({i:C}\u2032), measurement included, two-qubit gate count {i:N}{sub:2q}({i:C}\u2032) and ESP"));
C.push(E("ESP({i:C}\u2032) = \u220F{isub:g\u2208C\u2032} (1 \u2212 {i:\u03b5}{isub:g}) \u00b7 \u220F{isub:q} (1 \u2212 {i:\u03b5}{isub:q}{sup:ro}),", 1));
C.push(bodyNoIndent(S("where {i:\u03b5}{isub:g} is the calibrated error of physical gate {i:g} and {i:\u03b5}{isub:q}{sup:ro} the readout error of measured qubit {i:q} [@nishio]. We regress the vector {b:y} = [ln {i:d}, ln(1 + {i:N}{sub:2q}), ln(\u2212ln ESP)] and seek {i:f}{sub:\u03b8} minimising")));
C.push(E("\u2112(\u03b8) = \ud835\udd3c{isub:C,o,s} \u2016 {i:f}{sub:\u03b8}({i:C}, {i:o}) \u2212 {b:y}({i:T}{isub:o,s}({i:C}, {i:D})) \u2016,", 2));
C.push(bodyNoIndent(S("so the optimum is the seed-averaged cost and the seed-to-seed spread of {b:y} is an irreducible error that no predictor can remove. One model is trained per device; the device is implicit in the labels.")));

C.push(h2("B", "Heterogeneous Circuit Graph"));
C.push(P("Fig. 1 summarises the pipeline. {i:C} is first canonicalised without any device information into the basis {u, cx} with light single-qubit merging. Maximal runs of CX gates acting on the same qubit pair, possibly interleaved with single-qubit gates on those wires, are consolidated into one {i:block} node, mirroring the two-qubit block consolidation performed by the compiler. This reduces the mean graph size from {{sherbrooke_cx_mean}} CX gates to {{sherbrooke_blocks_mean}} block nodes. Each block {i:b} has ten features: log-multiplicity, log single-qubit gates inside the block and immediately before it on each wire, normalised ASAP layer indices, first-use flags of both wires and the summed and absolute-difference interaction degrees of its qubits. Each logical qubit {i:q} becomes a qubit node with six features: interaction degree and weighted degree, log single-qubit count, clustering coefficient, a flag for degree above three (the heavy-hex maximum, which forces SWAPs) and an idle flag. Directed {i:sequence} edges connect a block to the next block on each of its wires, and {i:incidence} edges connect a block to its two qubits. A 23-dimensional global descriptor {b:g} holds width, gate counts, logical depth, interaction-graph density, maximum degree and degree excess \u03a3{isub:q} max(0, deg{i:q} \u2212 3), mean shortest-path length and diameter, parallelism and a one-hot optimisation level."));

C.push(h2("C", "Structural Prior and Log-Overhead Targets"));
C.push(P("Routing overhead is approximately a ratio: the transpiled cost grows with its logical counterpart, and the ratio depends on the circuit\u2019s structure rather than on its size. We therefore predict residuals above a device-agnostic prior {b:b}({i:C}):"));
C.push(E("{b:b} = [ln({i:d}{isub:L} + 1), ln(1 + {i:N}{isub:cx}), ln({i:N}{isub:cx} + 2.5{i:n})],", 3));
C.push(E("{b:\u0177} = {b:b}({i:C}) + {i:f}{sub:\u03b8}({i:C}, {i:o}),", 4));
C.push(bodyNoIndent(S("where {i:d}{isub:L} and {i:N}{isub:cx} are the logical depth and CX count; the term 2.5{i:n} reflects that median readout errors are several times larger than median two-qubit gate errors on the studied devices (about 2.5\u00d7 on Eagle and 5.3\u00d7 on Heron, after removing faulty couplers). The same prior is given to every learned baseline, so it is not a source of unfair advantage.")));

C.push(figure(__dirname + "/../results/fig_arch.png", 236, FIG1_H));
C.push(figCaption(1, "Surrogate pipeline: the device-agnostic logical circuit is turned into a heterogeneous graph of two-qubit blocks and qubits, processed by HCG-Net together with global descriptors, and decoded as log-overheads above a structural prior."));
C.push(h2("D", "HCG-Net Architecture"));
C.push(P("Block, qubit and global inputs are embedded by two-layer MLPs into {i:h} = 64 dimensions, and the global embedding {b:c} = \u03c6{sub:g}({b:g}) is added to every node state. Each of {i:L} = 3 message-passing rounds [@gilmer] updates the block states from their predecessors \ud835\udcab({i:b}), successors \ud835\udcae({i:b}) and incident qubits \ud835\udcac({i:b}), and then the qubit states from their incident blocks \u212c({i:q}):"));
C.push(E("{b:m}{isub:b} = [{b:h}{isub:b} \u2225 \u0101{isub:\ud835\udcab(b)} \u2225 \u0101{isub:\ud835\udcae(b)} \u2225 \u0101{isub:\ud835\udcac(b)}],", 5));
C.push(E("{b:h}{isub:b} \u2190 LN({b:h}{isub:b} + \u03c8{isub:b}({b:m}{isub:b})),", 6));
C.push(E("{b:h}{isub:q} \u2190 LN({b:h}{isub:q} + \u03c8{isub:q}([{b:h}{isub:q} \u2225 \u0101{isub:\u212c(q)}])),", 7));
C.push(bodyNoIndent(S("where \u0101 denotes the mean of neighbour states, \u2225 concatenation, LN layer normalisation and \u03c8 two-layer SiLU MLPs. Readout concatenates mean and max pools over blocks and over qubits with {b:c}; mean and max are size-intensive, so extensive information enters only through {b:g} and the prior. A three-layer MLP head outputs {i:f}{sub:\u03b8}; targets are standardised per task and the three tasks are trained jointly with an equally weighted Huber loss (\u03b4 = 1). The network has 263 k parameters. We train for 60 epochs with AdamW [@adamw] (weight decay 10{sup:\u22124}), a one-cycle schedule with peak learning rate 3\u00d710{sup:\u22123}, batch size 64 and gradient clipping at 5, keeping the checkpoint with the lowest validation loss. {i:HCG-Ens} averages the log-space predictions of HCG-Net and of the gradient-boosted baseline (Section IV-C), which uses complementary inputs (engineered descriptors) and a different inductive bias. Graph construction is linear in the number of gates and inference costs {i:O}({i:L}(|{i:B}| + |{i:E}|){i:h}{sup:2}) for |{i:B}| blocks and |{i:E}| edges, independent of device size; Algorithm 1 summarises inference.")));
C.push(new Paragraph({ children: [], spacing: { before: 0, after: 60 } }));
C.push(algoBox(S("{b:Algorithm 1.} HCG surrogate inference for circuit {i:C} at optimisation level {i:o}"), [
  S("Transpile {i:C} to the basis {u, cx} without a device (light 1q merging)."),
  S("Scan gates in order; for a CX on ({i:c}, {i:t}): if the last block on wire {i:c} is also the last block on wire {i:t}, increment its multiplicity and absorb pending 1q gates; else open block {i:b}\u2032, add sequence edges from the last blocks on {i:c} and {i:t}, and incidence edges {i:b}\u2032\u2013{i:c}, {i:b}\u2032\u2013{i:t}."),
  S("Compute block and qubit features, global descriptor {b:g}({i:C}, {i:o}) and prior {b:b}({i:C}) by (3)."),
  S("Run {i:L} rounds of (5)\u2013(7); pool mean and max over blocks and qubits; apply the head to obtain {i:f}{sub:\u03b8}."),
    S("HCG-Ens averages {b:\u0177} with GBDT; return depth = e{sup:\u0177\u2081}, {i:N}{sub:2q} = e{sup:\u0177\u2082} \u2212 1, ESP = exp(\u2212e{sup:\u0177\u2083})."),
]));

// ------------------------------------------------------------------ IV. Setup
C.push(h1(4, "Experimental Setup"));
C.push(h2("A", "Devices, Compiler and Labels"));
C.push(P("We use the calibration snapshots shipped with qiskit-ibm-runtime for FakeSherbrooke (Eagle r3, 127 qubits, ECR) and FakeTorino (Heron r1, 133 qubits, CZ). Couplers flagged as non-operational (error \u2265 0.5) are removed and the target is restricted to its largest connected component (122 and 129 usable qubits), as on production systems. Each circuit is transpiled with the Qiskit 1.4.6 preset pass manager [@qiskit] at an optimisation level drawn uniformly from {1, 2, 3}; these levels use SABRE layout and routing with increasing numbers of trials, noise-aware VF2 post-layout and progressively stronger peephole and resynthesis passes. Labels are the depth, N{sub:2q} and ESP of (1) of the compiled circuit with final measurement of all qubits. For 5% of in-distribution circuits we repeat transpilation with five seeds to measure the noise floor. Software: Qiskit 1.4.6, qiskit-ibm-runtime 0.44.1, PyTorch 2.15 (CPU) and scikit-learn 1.9.1."));
C.push(h2("B", "Dataset"));
C.push(P("Nine families are generated with random parameters (Table II): random two-qubit-operand circuits, QFT with approximation degree 0 to {i:n}/2, GHZ with chain, star or tree fan-out, Bernstein\u2013Vazirani with random secrets, QAOA MaxCut on Erd\u0151s\u2013R\u00e9nyi, 3-regular, ring or grid graphs ({i:p} = 1\u20133) with shuffled commuting-term order, EfficientSU2 ans\u00e4tze with linear, circular, full or shifted-circular entanglement, quantum volume [@qv], ripple-carry and QFT adders, and Heisenberg Trotter circuits on rings and grids; the families mirror those of MQT Bench [@mqtbench] but are generated at controlled widths so that size and family hold-out can be studied. In-distribution (ID) widths are 4\u201320 qubits ({{sherbrooke_nid}} Eagle circuits) and the size-extrapolation (OOD) pool has 21\u201328 qubits ({{sherbrooke_nood}} circuits). Compiled depth reaches {{sherbrooke_depth_max}}, N{sub:2q} reaches {{sherbrooke_twoq_max}} and ESP spans from {{sherbrooke_esp_min}} to 0.98."));
C.push(...tableHead("II", "Eagle Dataset by Family (medians)"));
C.push(table(N.table1, [62, 22, 27, 30, 30, 34, 33]));
C.push(new Paragraph({ children: runs("CX{sub:L}: logical CX count; t: transpilation wall-clock time.", { size: 14 }), alignment: AlignmentType.LEFT, spacing: { before: 2 * PT, after: 6 * PT } }));
C.push(h2("C", "Baselines"));
C.push(P("A {i:scaling law} fits a ridge regression of each log-overhead on ln {i:d}{isub:L}, ln {i:N}{isub:cx}, ln {i:N}{isub:u}, ln {i:n} and the optimisation level. The remaining baselines use a 55-dimensional descriptor vector (the 23 global descriptors plus means and maxima of the block and qubit features): ridge regression, a random forest (300 trees), a two-layer MLP (256 units) and histogram gradient-boosted trees (GBDT; up to 800 iterations, learning rate 0.05, early stopping) [@sklearn]. All baselines predict the same log-overhead targets of (3)\u2013(4)."));
C.push(h2("D", "Protocol and Metrics"));
C.push(P("The ID pool is split 70/15/15 into training, validation and test sets with {{n_seeds}} different random seeds that also seed model initialisation; the OOD pool is a fixed test set. We report mean absolute percentage error (MAPE) for depth, N{sub:2q} (denominator max(1, {i:N}{sub:2q})) and \u2212ln ESP (the infidelity exponent), Spearman\u2019s \u03c1 and absolute error of ESP, and the recall of the true top-10% ESP circuits among the predicted top 10%. The noise floor is the MAPE of predicting one seed\u2019s label by the mean of the other four. All runs use two CPU cores and no GPU."));

// ------------------------------------------------------------------ V. Results
C.push(h1(5, "Results"));
C.push(h2("A", "In-Distribution Accuracy and the Noise Floor"));
C.push(...tableHead("III", "ID Test Accuracy on Eagle (MAPE %, mean\u00b1std over " + N.n_seeds + " splits); HCG-Net and HCG-Ens are ours"));
C.push(table(N.table2, [55, 45, 45, 45, 29, 24]));
C.push(P("Table III shows that learned surrogates predict all three costs accurately. HCG-Net reaches {{hcgnet_id_depth}}%, {{hcgnet_id_twoq}}% and {{hcgnet_id_esp}}% MAPE, with {{hcgnet_id_twoq_w10}}% of N{sub:2q} predictions and {{hcgnet_id_esp_w10}}% of \u2212ln ESP predictions within 10% of the label. Tree ensembles on our descriptors are strong competitors: the random forest and GBDT are within one standard deviation of HCG-Net on depth and \u2212ln ESP, and GBDT is better on N{sub:2q} ({{gbdt_id_twoq}}% versus {{hcgnet_id_twoq}}%). HCG-Ens, which averages HCG-Net and GBDT, is the most accurate model on every target ({{hcg_ens_id_depth}}%, {{hcg_ens_id_twoq}}% and {{hcg_ens_id_esp}}%), with ESP rank correlation \u03c1 = {{hcg_ens_id_rho}}, a mean absolute ESP error of {{hcg_ens_id_espmae}} and {{hcg_ens_id_twoq_w10}}% of N{sub:2q} predictions within 10%. On the seed-0 split, paired Wilcoxon signed-rank tests on per-circuit errors (945 circuits) confirm that HCG-Ens is more accurate than GBDT on all three targets (p \u2264 {{sig_id_ens_gbdt_pmax}}; depth MAPE difference {{sig_id_hcg_ens_vs_gbdt_depth_diff}} points, 95% bootstrap CI {{sig_id_hcg_ens_vs_gbdt_depth_ci}}), whereas HCG-Net and GBDT do not differ at the 1% level on any target (smallest p = {{sig_id_net_gbdt_pmin}}). A scaling law that ignores circuit structure is 4\u20138\u00d7 worse, which confirms that routing overhead depends strongly on the interaction structure and not only on gate counts."));
C.push(figure(__dirname + "/../results/fig_parity.png", 236, FIG3_H));
C.push(figCaption(2, "Predicted versus transpiled cost for HCG-Ens on the seed-0 split (log axes; diagonal = perfect prediction)."));
C.push(P("Predictions follow the labels over three to four orders of magnitude (Fig. 2). On the seed-0 split the HCG-Ens error varies little with optimisation level (depth {{ol_min_depth}}\u2013{{ol_max_depth}}%, N{sub:2q} {{ol_min_twoq}}\u2013{{ol_max_twoq}}%) and more across families (depth {{fam_min_depth}}% for {{fam_min_depth_name}} to {{fam_max_depth}}% for {{fam_max_depth_name}})."));
C.push(P("These errors must be read against the compiler\u2019s stochasticity. Over {{nf_sherbrooke_n}} circuits transpiled with five seeds, the mean coefficient of variation is {{cv_sherbrooke_depth}}% for depth, {{cv_sherbrooke_twoq}}% for N{sub:2q} and {{cv_sherbrooke_esp}}% for \u2212ln ESP, and the leave-one-seed-out floor is {{nf_sherbrooke_depth}}%, {{nf_sherbrooke_twoq}}% and {{nf_sherbrooke_esp}}%. Because each training label is itself a single-seed draw, HCG-Ens operates within {{ratio_hcg_ens_depth}}\u00d7, {{ratio_hcg_ens_twoq}}\u00d7 and {{ratio_hcg_ens_esp}}\u00d7 of this floor. Depth is the noisiest target, plausibly because SABRE trials are scored mainly by SWAP count rather than by depth [@lightsabre]."));

C.push(h2("B", "Extrapolation to Wider Circuits"));
C.push(...tableHead("IV", "Size-OOD Test (21\u201328 qubits) on Eagle (MAPE %)"));
C.push(table(N.table3, [58, 47, 47, 47, 42]));
C.push(figure(__dirname + "/../results/fig_size.png", 236, FIG2_H));
C.push(figCaption(3, S("MAPE versus logical width on the seed-0 split (ID test for 4\u201320 qubits, OOD pool for 21\u201328). Dashed: five-seed noise floor.")));
C.push(P("Errors grow when the models extrapolate to 21\u201328 qubits (Table IV, Fig. 3). Among single models HCG-Net extrapolates best on depth ({{hcgnet_ood_depth}}% versus {{gbdt_ood_depth}}% for GBDT), whereas GBDT degrades less on N{sub:2q} and \u2212ln ESP. The ID error is nearly flat in width, so the jump is caused by extrapolation rather than by wider circuits being intrinsically harder. On the seed-0 split both models are unbiased in distribution (mean log-error below 0.005 in magnitude) but under-predict all three costs on the OOD pool (\u2212ln ESP: \u22120.073 for HCG-Net, \u22120.039 for GBDT), plausibly because wider circuits use more of the chip, including its worse qubits and couplers, which a device-agnostic input cannot see. HCG-Ens has the lowest OOD MAPE on every target ({{hcg_ens_ood_depth}}%, {{hcg_ens_ood_twoq}}% and {{hcg_ens_ood_esp}}%), significantly below GBDT on all three (paired Wilcoxon, p \u2264 {{sig_ood_ens_gbdt_pmax}}; depth difference {{sig_ood_hcg_ens_vs_gbdt_depth_diff}} points, CI {{sig_ood_hcg_ens_vs_gbdt_depth_ci}}). The MLP nevertheless has the lowest absolute ESP error ({{mlp_ood_espmae}}), because MAPE on \u2212ln ESP weights low-fidelity circuits more heavily."));

C.push(h2("C", "Ablation"));
C.push(...tableHead("V", "Ablation on Eagle (MAPE %, mean of " + N.abl_nseed + " splits)"));
C.push(table(N.table4, [78, 27, 27, 27, 27, 27, 27], { headRows: 2, spans: { "0,1": 3, "0,4": 3 }, skip: { "0,2": 1, "0,3": 1, "0,5": 1, "0,6": 1 } }));
C.push(new Paragraph({ children: runs("\u2020 Seed 0 only, ID test only.", { size: 14 }), alignment: AlignmentType.LEFT, spacing: { before: 2 * PT, after: 6 * PT } }));
C.push(P("Table V isolates the design choices (same splits and seeds as Table III). The structural prior matters most for extrapolation: without it HCG-Net\u2019s OOD error rises from {{abl_full_ood_twoq}}% to {{abl_no_prior_ood_twoq}}% for N{sub:2q} and from {{abl_full_ood_esp}}% to {{abl_no_prior_ood_esp}}% for \u2212ln ESP, while ID accuracy changes much less; for GBDT it helps N{sub:2q} and \u2212ln ESP but slightly hurts depth extrapolation. Global descriptors are indispensable in distribution (ID error {{abl_no_glob_id_depth}}%, {{abl_no_glob_id_twoq}}% and {{abl_no_glob_id_esp}}% without them), partly because they carry the optimisation level, yet removing them improves \u2212ln ESP extrapolation, suggesting that extensive descriptors encourage width-dependent offsets. Sequence edges do not help in distribution but cut OOD depth error from {{abl_no_seq_ood_depth}}% to {{abl_full_ood_depth}}%, as expected for a critical-path quantity; qubit nodes lower OOD N{sub:2q} error from {{abl_no_qubit_ood_twoq}}% to {{abl_full_ood_twoq}}%; sum pooling degrades all three OOD errors. Three single-task networks are not better overall (seed 0: {{abl_single_task_id_depth}}%, {{abl_single_task_id_twoq}}% and {{abl_single_task_id_esp}}% versus {{hcgnet_s0_id_depth}}%, {{hcgnet_s0_id_twoq}}% and {{hcgnet_s0_id_esp}}%) at three times the training cost."));

C.push(h2("D", "Unseen Circuit Families"));
C.push(...tableHead("VI", "Leave-One-Family-Out on Eagle (MAPE %)"));
C.push(table(N.table5 || [["pending"]], [66, 29, 29, 29, 29, 29, 29], { headRows: 2, spans: { "0,1": 2, "0,3": 2, "0,5": 2 }, skip: { "0,2": 1, "0,4": 1, "0,6": 1 } }));
C.push(new Paragraph({ children: runs("HCG: HCG-Net. Test: all widths (4\u201328) of the held-out family; bold: better model.", { size: 14 }), alignment: AlignmentType.LEFT, spacing: { before: 2 * PT, after: 6 * PT } }));
C.push(P("Holding out a whole family is the hardest test (Table VI): mean errors are 2.6\u20133.2\u00d7 the ID values. Here the graph model is clearly better: HCG-Net has the lower error in {{lofo_wins}} of {{lofo_cells}} family\u2013target cells and reduces mean MAPE from {{lofo_mean_gbdt_depth}}%, {{lofo_mean_gbdt_twoq}}% and {{lofo_mean_gbdt_esp}}% (GBDT) to {{lofo_mean_hcgnet_depth}}%, {{lofo_mean_hcgnet_twoq}}% and {{lofo_mean_hcgnet_esp}}%, with the largest gains on structurally distinctive families such as Heisenberg Trotter (depth {{lofo_trotter_gbdt_depth}}% versus {{lofo_trotter_hcgnet_depth}}%) and quantum volume ({{lofo_qv_gbdt_depth}}% versus {{lofo_qv_hcgnet_depth}}%), whose descriptors fall outside the training range while block-level message passing still applies. GBDT wins the other eight cells by at most 4.7 points. The ensemble lies in between (mean {{lofo_mean_hcg_ens_depth}}%, {{lofo_mean_hcg_ens_twoq}}%, {{lofo_mean_hcg_ens_esp}}%), so for new families HCG-Net alone is preferable for depth."));

C.push(h2("E", "Second Device and Transfer"));
C.push(...tableHead("VII", "Heron (FakeTorino) ID Test and Transfer (MAPE %, " + N.tor_nseed + " splits)"));
C.push(table(N.table6 || [["pending"]], [100, 47, 47, 47]));
C.push(new Paragraph({ children: runs("Fine-tune: Eagle-trained HCG-Net, 40 epochs, peak learning rate 10{sup:\u22123}.", { size: 14 }), alignment: AlignmentType.LEFT, spacing: { before: 2 * PT, after: 6 * PT } }));
C.push(P("On Heron (Table VII; {{tor_ntrain}} training circuits per split) HCG-Ens is again most accurate on depth ({{tor_hcg_ens_id_depth}}%) and \u2212ln ESP ({{tor_hcg_ens_id_esp}}%), but HCG-Net alone trails GBDT on N{sub:2q} ({{tor_hcgnet_id_twoq}}% versus {{tor_gbdt_id_twoq}}%), so with less than half the Eagle data the graph model is the more data-hungry learner; on wider Heron circuits it again extrapolates depth far better ({{tor_hcgnet_ood_depth}}% versus {{tor_gbdt_ood_depth}}%). Applied zero-shot, the Eagle model fails on depth ({{tf_zero_shot_depth}}%) and \u2212ln ESP ({{tf_zero_shot_esp}}%) because native gates (ECR versus CZ) and error rates differ, yet reaches {{tf_zero_shot_twoq}}% on N{sub:2q}, consistent with both chips being heavy-hex. Fine-tuning on 100 Heron circuits lowers depth and \u2212ln ESP error from {{tf_scratch_100_depth}}% and {{tf_scratch_100_esp}}% (scratch) to {{tf_finetune_100_depth}}% and {{tf_finetune_100_esp}}%, whereas for N{sub:2q} the zero-shot model already matches training from scratch on all data ({{tf_scratch_all_twoq}}%) and only full fine-tuning improves it ({{tf_finetune_all_twoq}}%). Routing knowledge transfers across heavy-hex devices; gate-set and calibration knowledge must be relearned."));

C.push(h2("F", "Inference Cost"));
C.push(P("On one CPU thread, building the graph takes {{t_feat_ms}} ms and an HCG-Net forward pass {{t_gnn1_ms}} ms per circuit at batch size 1 or {{t_gnnb_ms}} ms per circuit in batches of 256. Preset transpilation of the same test circuits, timed during dataset generation with two parallel workers, took {{t_transpile_ms}} ms on average (median {{t_transpile_med_ms}} ms; {{t_ol1_ms}}, {{t_ol2_ms}} and {{t_ol3_ms}} ms at levels 1, 2 and 3), so the surrogate is about {{speedup1}}\u00d7 faster per query and {{speedupb}}\u00d7 faster in batch mode, while replacing a multi-seed estimate with a single seed-averaged prediction. Training takes about {{hcgnet_train_s}} s (HCG-Net) and {{gbdt_train_s}} s (GBDT) per split."));

// ------------------------------------------------------------------ VI. Discussion & limitations
C.push(h1(6, "Discussion and Limitations"));
C.push(P("Three lessons emerge. First, in distribution most of the predictable variance is captured by well-chosen circuit-level descriptors, so a tuned tree ensemble is a strong, cheap baseline that learned compiler models should be compared against; HCG-Net alone is not significantly better there. Second, graph learning pays off where structure must generalise: HCG-Net wins {{lofo_wins}} of {{lofo_cells}} family\u2013target cells on unseen families and extrapolates depth better on both devices, and because the two learners err differently their ensemble is the most accurate model on random splits and wider circuits. Third, compiler stochasticity bounds any surrogate: the best model is within {{ratio_hcg_ens_depth}}\u2013{{ratio_hcg_ens_esp}}\u00d7 of the seed-noise floor, so predicting the distribution or best-of-{i:k} cost is a more useful next target than further ID gains. For screening, a top-10% ESP recall of {{hcg_ens_top10}}% lets the surrogate shortlist candidates before a few real compilations."));
C.push(P("The study has clear limits. Labels come from static calibration snapshots of simulated devices; real calibrations drift daily, so a deployed surrogate must be retrained cheaply (Section V-E suggests this is feasible) or take calibration data as input. ESP is a first-order fidelity proxy [@nishio] that ignores crosstalk and idle decoherence, and its correlation with measured execution quality is weaker than often assumed [@hopf25]. We studied one compiler version and widths up to 28 qubits on two devices, and LOFO and the paired significance tests use a single split. The surrogate predicts the seed-averaged cost, whereas some uses need the distribution or best-of-{i:k} cost."));

// ------------------------------------------------------------------ VII. Conclusion
C.push(h1(7, "Conclusion"));
C.push(P("We framed transpilation-cost prediction as noise-floor-aware multi-target regression and proposed HCG-Net, a heterogeneous graph network over consolidated two-qubit blocks and logical qubits that predicts post-routing depth, two-qubit gate count and ESP as log-overheads above a structural prior. On {{sherbrooke_count}} Eagle and {{torino_count}} Heron circuits labelled by the Qiskit preset pass managers, the HCG-Net/GBDT ensemble reaches {{hcg_ens_id_depth}}%, {{hcg_ens_id_twoq}}% and {{hcg_ens_id_esp}}% MAPE, within about 1.2\u20131.3\u00d7 of the compiler\u2019s own seed-to-seed variability, and answers {{speedup1}}\u2013{{speedupb}}\u00d7 faster than transpilation. The structural prior is essential for extrapolation, graph structure helps most on unseen families, and routing knowledge transfers between heavy-hex devices. Future work will condition the model on calibration graphs, predict cost distributions over seeds, and validate fidelity predictions on hardware."));
C.push(h1(0, "Code and Data Availability"));
C.push(P("Two archives accompany this manuscript. The code archive contains the circuit generator, graph construction, HCG-Net and baseline implementations, all experiment, analysis and figure scripts, the raw result files and the script that derives every number and table in this paper from them, the logs of the reported runs and the trained seed-0 HCG-Net weights for both devices. The data archive contains the {{sherbrooke_count}} Eagle and {{torino_count}} Heron logical circuits in QPY format with their labels (depth, N{sub:2q}, ln ESP, transpilation time, seed and the extra-seed labels of the noise-floor subset) as CSV, together with an import script that rebuilds the training inputs; retraining from the released data reproduces the reported seed-0 results exactly on the stated software versions. Both archives will be deposited in a public repository with a persistent identifier upon acceptance."));
C.push(h1(0, "Acknowledgment"));
C.push(P("The author used an AI assistant (Claude, Anthropic) to help write the experiment code and draft the manuscript; all experiments were executed and all numbers are taken from the logged results, and the author reviewed the content and takes responsibility for it."));

// ------------------------------------------------------------------ References
C.push(h1(0, "References"));
ORDER.forEach((k, i) => C.push(ref(i + 1, REFS[k])));
const unused = Object.keys(REFS).filter((k) => !ORDER.includes(k));
if (unused.length) console.warn("uncited references:", unused.join(", "));

const final = C;

// ------------------------------------------------------------------ document
const footerFirst = new Footer({ children: [new Paragraph({ children: [new TextRun({ text: "XXX-X-XXXX-XXXX-X/XX/$XX.00 \u00a920XX IEEE", font: FONT, size: 16 })], alignment: AlignmentType.LEFT })] });
const page = { size: { width: Math.round(595.3 * PT), height: Math.round(841.9 * PT) } };
const doc = new Document({
  creator: "Gousia Thahniyath", title: TITLE,
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [
    { properties: { page: { ...page, margin: { top: 27 * PT, bottom: 72 * PT, left: Math.round(44.65 * PT), right: Math.round(44.65 * PT), header: 36 * PT, footer: 36 * PT } }, titlePage: true },
      footers: { first: footerFirst, default: new Footer({ children: [new Paragraph({ children: [] })] }) }, children: titleBlock },
    { properties: { type: SectionType.CONTINUOUS, page: { ...page, margin: { top: Math.round(22.5 * PT), bottom: 72 * PT, left: Math.round(44.65 * PT), right: Math.round(44.65 * PT), header: 36 * PT, footer: 36 * PT } } },
      children: authorBlock },
    { properties: { type: SectionType.CONTINUOUS, column: { count: 2, space: 18 * PT, equalWidth: true }, page: { ...page, margin: { top: 54 * PT, bottom: 72 * PT, left: Math.round(45.35 * PT), right: Math.round(45.35 * PT), header: 36 * PT, footer: 36 * PT } } },
      children: final },
  ],
});
module.exports = { doc };
if (require.main === module) {
  Packer.toBuffer(doc).then((b) => { fs.writeFileSync(process.argv[2] || "paper.docx", b); console.log("written"); });
}
